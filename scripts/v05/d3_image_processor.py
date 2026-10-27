"""Static-image PIL adaptation; preserves pinned D3 resize and tokenizer contract."""
import json
from pathlib import Path
from transformers import AutoTokenizer, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

class StaticQwen3VLProcessor(Qwen3VLProcessor):
    """Disable unavailable video modality while keeping native image processing."""
    def check_argument_for_proper_class(self, name, argument):
        if name == "video_processor" and argument is None:
            return type(None)
        return super().check_argument_for_proper_class(name, argument)

    def __call__(self, *args, **kwargs):
        if kwargs.get("videos") is not None:
            raise ValueError("exp87 adapter supports static images only")
        return super().__call__(*args, **kwargs)

def load_processor(root):
    root=Path(root)
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
    image=Qwen2VLImageProcessorPil.from_pretrained(root,local_files_only=True)
    tokenizer.padding_side='left'
    image.size={'shortest_edge':65536,'longest_edge':1638400}
    chat=(root/'chat_template.jinja').read_text() if (root/'chat_template.jinja').exists() else None
    return StaticQwen3VLProcessor(image_processor=image,tokenizer=tokenizer,video_processor=None,chat_template=chat)

if __name__=='__main__':
    from PIL import Image
    root=Path('runs/exp87-d3-ui/snapshot.txt').read_text()
    processor=load_processor(root)
    sample=json.loads(Path('data/v05/exp85/ui_dev.jsonl').read_text().splitlines()[0])
    text='<|vision_start|><|image_pad|><|vision_end|>Choose A or B.'
    x=processor(text=[text],images=[Image.open(Path('data')/sample['image']).convert('RGB')],return_tensors='pt')
    import torch
    config=json.loads((Path(root)/'config.json').read_text())
    image_id=config['image_token_id']
    expected=int(x.image_grid_thw.prod())//processor.image_processor.merge_size**2
    assert int((x.input_ids==image_id).sum())==expected
    assert int(x.input_ids.min())>=0 and int(x.input_ids.max())<config['text_config']['vocab_size']
    print('PASS PIL processor, token range:',int(x.input_ids.min()),int(x.input_ids.max()),'image tokens:',expected,'pixel values:',list(x.pixel_values.shape))
