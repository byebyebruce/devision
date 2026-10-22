"""Let a deVision model play Snake: each step the board is drawn as a picture and the model picks a direction.

    uv run python scripts/demo/snake.py --checkpoint runs/r21-soft/train [--games 5] [--max-steps 300] [--device cuda]
    uv run python scripts/demo/snake.py --policy random --games 5

10 x 10 board, the snake starts with length 3 in the middle, one apple at a time. Each step the model gets the picture
and a choice question over the moves that do not reverse into its own neck ("up" / "down" / "left" / "right", in a
shuffled order so position bias does not decide); the highest-probability move is taken. A game ends when the snake
hits a wall or itself, or after --max-steps. Prints steps survived and apples eaten per game. A demo, not an
evaluation set; nothing here is trained on.
"""
import argparse
import random
from typing import List, Tuple

from PIL import Image, ImageDraw

N = 10
CELL = 25
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
QUESTION = ("This is a game of Snake. The green snake's head is the dark green square. Which way should the snake "
            "move next to get closer to the red apple without hitting a wall or its own body?")


def draw(snake: List[Tuple[int, int]], apple: Tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", (N * CELL, N * CELL), "white")
    d = ImageDraw.Draw(img)
    for i in range(N + 1):
        d.line([(i * CELL, 0), (i * CELL, N * CELL)], fill="#dddddd")
        d.line([(0, i * CELL), (N * CELL, i * CELL)], fill="#dddddd")
    d.rectangle([0, 0, N * CELL - 1, N * CELL - 1], outline="black", width=3)
    ax, ay = apple
    d.ellipse([ax * CELL + 3, ay * CELL + 3, (ax + 1) * CELL - 3, (ay + 1) * CELL - 3], fill="red")
    for k, (x, y) in enumerate(snake):
        d.rectangle([x * CELL + 2, y * CELL + 2, (x + 1) * CELL - 2, (y + 1) * CELL - 2],
                    fill="#0b6623" if k == 0 else "#5cd65c")
    return img


def legal(snake) -> List[str]:
    hx, hy = snake[0]
    nx, ny = snake[1]
    return [m for m, (dx, dy) in MOVES.items() if (hx + dx, hy + dy) != (nx, ny)]


def play(choose, rng: random.Random, max_steps: int) -> Tuple[int, int, str]:
    snake = [(5, 5), (4, 5), (3, 5)]
    free = lambda: rng.choice([(x, y) for x in range(N) for y in range(N) if (x, y) not in snake])  # noqa: E731
    apple = free()
    apples = 0
    for step in range(1, max_steps + 1):
        opts = legal(snake)
        rng.shuffle(opts)
        m = choose(draw(snake, apple), opts)
        dx, dy = MOVES[m]
        head = (snake[0][0] + dx, snake[0][1] + dy)
        if not (0 <= head[0] < N and 0 <= head[1] < N):
            return step - 1, apples, "wall"
        grow = head == apple
        body = snake if grow else snake[:-1]
        if head in body:
            return step - 1, apples, "self"
        snake = [head] + body
        if grow:
            apples += 1
            if len(snake) == N * N:
                return step, apples, "board full"
            apple = free()
    return max_steps, apples, "max steps"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--policy", choices=["model", "random"], default="model")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.policy == "model":
        import devision
        model = devision.load(a.checkpoint, device=a.device)

        def choose(img, opts):
            r = model.predict(image=img, state="", questions={
                "move": {"type": "choice", "instructions": QUESTION, "criteria": {o: None for o in opts}}})
            probs = r["answers"]["move"]["probabilities"]
            return max(probs, key=lambda o: probs[o])
    else:
        pick = random.Random(a.seed + 1)

        def choose(img, opts):
            return pick.choice(opts)
    results = []
    for g in range(a.games):
        steps, apples, end = play(choose, random.Random(a.seed * 1000 + g), a.max_steps)
        results.append((steps, apples))
        print("game %d: %d steps, %d apples, ended by %s" % (g + 1, steps, apples, end), flush=True)
    print("mean: %.1f steps, %.1f apples over %d games" % (sum(s for s, _ in results) / len(results),
                                                          sum(x for _, x in results) / len(results), len(results)))


if __name__ == "__main__":
    main()
