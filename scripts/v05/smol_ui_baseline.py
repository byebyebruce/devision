"""Pinned SmolVLM single-step letter readout on the existing UI development panel."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, Idefics3Processor
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil

WORK = Path(os.environ.get('DEVISION_BASELINE_OUT', 'runs/exp86-smol-ui-baseline'))
REPO = 'HuggingFaceTB/SmolVLM-256M-Instruct'
REV = '7e3e67edbbed1bf9888184d9df282b700a323964'


def status(state, **kwargs):
    p = WORK / 'STATUS.tmp'
    p.write_text(json.dumps(dict(state=state, updated=time.strftime('%F %T %z'), **kwargs), indent=2)+'\n')
    p.replace(WORK / 'STATUS.json')


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'STARTED').open('x').close()
    start = time.monotonic()
    status('downloading', repo=REPO, revision=REV)
    path = snapshot_download(REPO, revision=REV, allow_patterns=['*.json', '*.txt', '*.safetensors'],
                             ignore_patterns=['onnx/*'], max_workers=2)
    status('loading', snapshot=path)
    image_processor = Idefics3ImageProcessorPil.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    chat = json.loads((Path(path) / 'chat_template.json').read_text())['chat_template']
    processor_config = json.loads((Path(path) / 'processor_config.json').read_text())
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    image_seq_len = processor_config['image_seq_len']
    expected = (config.vision_config.image_size // config.vision_config.patch_size // config.scale_factor) ** 2
    assert image_seq_len == expected, (image_seq_len, expected)
    processor = Idefics3Processor(image_processor=image_processor, tokenizer=tokenizer,
                                 chat_template=chat, image_seq_len=image_seq_len)
    assert processor.image_token_id == config.image_token_id
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {'longest_edge': 512}
    model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    letters = ['A', 'B', 'C', 'D']
    token_ids = [processor.tokenizer.encode(x, add_special_tokens=False) for x in letters]
    assert all(len(x) == 1 for x in token_ids), token_ids
    token_ids = [x[0] for x in token_ids]
    identity = dict(repo=REPO, revision=REV, snapshot=path, torch=torch.__version__,
                    precision='bfloat16', device=torch.cuda.get_device_name(),
                    readout='last prompt token logits over A/B/C/D; no generation; T=1',
                    image_policy='one tile; no splitting; longest_edge=512',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    processor=processor.image_processor.to_dict(), token_ids=token_ids,
                    image_seq_len=image_seq_len, image_token_id=processor.image_token_id,
                    text_vocab_size=config.text_config.vocab_size)
    (WORK / 'IDENTITY.json').write_text(json.dumps(identity, indent=2)+'\n')
    identity['assistant_prefix'] = os.environ.get('DEVISION_ASSISTANT_PREFIX', '')
    identity['source_override'] = os.environ.get('DEVISION_BASELINE_SOURCE')
    (WORK / 'IDENTITY.json').write_text(json.dumps(identity, indent=2)+'\n')
    controls = os.environ.get('DEVISION_BASELINE_CONTROLS', 'real,mismatched').split(',')
    assert controls and all(c in {'real', 'mismatched', 'reversed'} for c in controls)
    for control in controls:
        filename = 'ui_dev_rotated.jsonl' if control == 'mismatched' else 'ui_dev.jsonl'
        source = Path(os.environ.get('DEVISION_BASELINE_SOURCE', str(Path('data/v05/exp85') / filename)))
        samples = list(map(json.loads, source.read_text().splitlines()))
        if control == 'mismatched' and os.environ.get('DEVISION_BASELINE_SOURCE'):
            groups = {}
            for sample in samples:
                groups.setdefault(sample['counterfactual_group'], []).append(sample)
            samples = []
            for group in groups.values():
                assert len(group) == 4
                assert len({tuple(x['questions']['q']['criteria']) for x in group}) == 1
                for j, sample in enumerate(group):
                    samples.append(dict(sample, image=group[(j+1) % 4]['image']))
        records = []
        with (WORK / f'ui.{control}.details.jsonl').open('x') as output:
            for i, sample in enumerate(samples):
                if time.monotonic()-start > 2400:
                    raise TimeoutError('40 minute download/loading/evaluation budget')
                q = sample['questions']['q']; options = list(q['criteria'])
                if control == 'reversed':
                    options.reverse()
                assert len(options) == 4
                prompt = q['instructions']+'\n'+'\n'.join(f'{l}. {o}' for l,o in zip(letters, options))+'\nAnswer with only the letter of the correct option.'
                messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
                text = processor.apply_chat_template(messages, add_generation_prompt=True) + os.environ.get('DEVISION_ASSISTANT_PREFIX', '')
                with Image.open(Path('data') / sample['image']) as image:
                    inputs = processor(text=text, images=[image.convert('RGB')], return_tensors='pt')
                assert 0 <= inputs['input_ids'].min().item()
                assert inputs['input_ids'].max().item() < config.text_config.vocab_size
                assert (inputs['input_ids'] == config.image_token_id).sum().item() == expected
                inputs = inputs.to('cuda')
                torch.cuda.synchronize(); tick = time.perf_counter()
                with torch.inference_mode():
                    result = model(**inputs, use_cache=False)
                    probabilities = result.logits[0, -1, token_ids].float().softmax(-1).cpu().tolist()
                torch.cuda.synchronize(); latency = (time.perf_counter()-tick)*1000
                assert all(math.isfinite(x) for x in probabilities)
                gold = sample['gold']['q']['probabilities']; gold_answer = max(gold, key=gold.get)
                predicted = options[max(range(4), key=probabilities.__getitem__)]
                row = dict(sample_id=sample['id'], qid='q', image_id=sample['image_id'],
                           image_used=sample['image'], source=sample['source'], type='choice',
                           options=options, gold=gold, gold_answer=gold_answer,
                           probabilities=dict(zip(options, probabilities)), prediction=predicted,
                           correct=predicted == gold_answer, nll=-math.log(max(probabilities[options.index(gold_answer)], 1e-30)),
                           brier=sum((p-gold[o])**2 for o,p in zip(options, probabilities)),
                           control=control, latency_ms=latency, input_tokens=inputs['input_ids'].shape[1])
                output.write(json.dumps(row)+'\n'); output.flush(); records.append(row)
                if i % 16 == 0:
                    status('evaluating', control=control, completed=i+1, total=len(samples))
                    print(control, i+1, 'accuracy', sum(r['correct'] for r in records)/len(records), flush=True)
        summary = dict(n=len(records), accuracy=sum(r['correct'] for r in records)/len(records),
                       nll=sum(r['nll'] for r in records)/len(records), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       timing_scope='model forward only; batch1; includes cold first call; not deployment latency')
        (WORK / f'ui.{control}.json').write_text(json.dumps(summary, indent=2)+'\n')
    status('completed', scope='UI development only; other panels and training pending', controls=controls)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if WORK.exists():
            status('execution_failed', error=repr(exc))
        raise
