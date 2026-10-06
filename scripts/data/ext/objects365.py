"""Extension pack, Objects365 (everyday photos with exhaustive boxes over 365 classes). Annotations CC BY 4.0,
academic use only; the pictures may not be redistributed (kept under data/, which is not in git).

    uv run python scripts/data/ext/objects365.py --root data [--images 54000] [--patches 0,1] [--per-picture 3] [--force]

Route: the raw Objects365 v2 training annotations (jxu124/objects365 on Hugging Face: the official boxes as parquet,
one row per picture, xyxy boxes with iscrowd / isfake / isreflected flags; 1.6 GB) plus only the needed picture
archives (guozonghao96/objects365: the official train patch<k>.tar.gz files; the v1 patches 0-15 hold about 34,700
pictures each in about 3.8 GB). Exhaustive boxes give true negatives, exact counts and left / right from box
centres, which FineVision's objects365_qa (sentence answers, about 200 GB) cannot. The picture of every annotation
row is matched by its file name inside the archive.

Pictures: those of --patches, in a deterministic order (by id hash), the first --images that yield a question;
photos within NEAR_BITS of a held-out photo are dropped (common.HeldOut.near_held). Questions per picture:
  - exist (noul): "Is there a <thing> ...?" Yes: a class whose largest real box (not a reflection, not fake)
    covers >= 1% of the picture. No: a class absent from the picture's exhaustive annotation, drawn by how often it
    co-occurs with the picture's classes (plausible), never a class that shares a confusion group or a head word
    with a class in the picture (no "car?" next to an SUV, no "traffic sign?" next to a stop sign);
  - count (choice, 0..10, common.count_options): only a class all of whose boxes are real, not crowd and >= 1% of
    the picture (a crowd box or tiny instances make the count unknowable); 0 = a plausible absent class as above;
  - position (choice left / right): "Is the X to the left or right of the Y?" for two classes with exactly one box
    each (>= 1%, real, not crowd), box centres more than 20% of the width apart, not confusable with each other.
Balance: exist yes = no per class; counts flattened; left / right equalised per unordered class pair (so the
wording alone says nothing), then at most --per-picture questions per picture and the exist balance redone.
Class names: Objects365's labels have typos and slashes ("Moniter/TV", "Bakset"); NAMES gives the English
singular / plural; vague classes ("Other Shoes", "Toiletry") are never asked, mass nouns and pairs are not counted.
"""
import hashlib
import os
import random
import re
import sys
import tarfile
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, balance_yes_no, base_args, choice,  # noqa: E402
                    count_options, finish, flatten_answers, hf_file, hf_files, noul, save_bytes)

SOURCE = "objects365"
ANN_REPO = "jxu124/objects365"
IMG_REPO = "guozonghao96/objects365"
MIN_AREA = 0.01      # a box must cover >= 1% of the picture to count as visible at 256 px
MIN_GAP = 0.20       # left / right: box centres at least this share of the width apart
MAX_COUNT = 10

# raw label | singular (an article is added; "!" = mass noun, no article) | plural for counting ("-" = not counted)
# Raw labels not listed are never asked.
_TABLE = """
Person|person|people
Cabinet/shelf|cabinet or shelf|-
Desk|desk|desks
Plate|plate|plates
Cup|cup|cups
Bottle|bottle|bottles
Bowl/Basin|bowl|bowls
Sink|sink|sinks
Faucet|faucet|faucets
Chair|chair|chairs
Lamp|lamp|lamps
Storage box|storage box|storage boxes
Bakset|basket|baskets
Moniter/TV|monitor or TV|monitors or TVs
Towel|towel|towels
Hat|hat|hats
Glasses|pair of glasses|-
Mirror|mirror|mirrors
Tomato|tomato|tomatoes
Power outlet|power outlet|power outlets
Knife|knife|knives
Speaker|speaker|speakers
Handbag/Satchel|handbag|handbags
Fork|fork|forks
Barrel/bucket|barrel or bucket|-
Bread|!bread|-
Trash bin Can|trash can|trash cans
Pen/Pencil|pen or pencil|-
Mouse|computer mouse|-
Keyboard|keyboard|keyboards
Picture/Frame|picture frame|picture frames
Book|book|books
Spoon|spoon|spoons
Car|car|cars
Bracelet|bracelet|bracelets
Apple|apple|apples
Stool|stool|stools
Sneakers|sneaker|-
Oven|oven|ovens
Carpet|carpet|-
Pot|pot|pots
Gas stove|gas stove|-
Toilet|toilet|toilets
Laptop|laptop|laptops
Pepper|pepper|peppers
Banana|banana|bananas
Street Lights|street light|street lights
Cell Phone|cell phone|cell phones
Orange/Tangerine|orange|oranges
Carrot|carrot|carrots
Lemon|lemon|lemons
Dinning Table|dining table|dining tables
Toilet Paper|!toilet paper|-
Broccoli|!broccoli|-
Watch|watch|watches
Bathtub|bathtub|bathtubs
Refrigerator|refrigerator|refrigerators
Microphone|microphone|microphones
Slippers|slipper|-
Potted Plant|potted plant|potted plants
Cutting/chopping Board|cutting board|cutting boards
Microwave|microwave|microwaves
Flower|flower|-
Telephone|telephone|telephones
Cucumber|cucumber|cucumbers
Pumpkin|pumpkin|pumpkins
Computer Box|computer tower|computer towers
Drum|drum|drums
Potato|potato|potatoes
Extention Cord|extension cord|-
Wine Glass|wine glass|wine glasses
Extractor|range hood|-
Vase|vase|vases
Onion|onion|onions
Blackboard/Whiteboard|blackboard or whiteboard|-
Bench|bench|benches
Guitar|guitar|guitars
Ring|ring|rings
Cymbal|cymbal|cymbals
Pillow|pillow|pillows
Strawberry|strawberry|strawberries
Bed|bed|beds
Cabbage|cabbage|cabbages
Napkin|napkin|napkins
Tripod|tripod|tripods
Couch|couch|couches
Toothbrush|toothbrush|toothbrushes
Tissue|tissue|-
Van|van|vans
Rice|!rice|-
Duck|duck|ducks
Truck|truck|trucks
Boat|boat|boats
Hanger|hanger|hangers
Remote|remote control|remote controls
Bicycle|bicycle|bicycles
Machinery Vehicle|construction vehicle|construction vehicles
Washing Machine/Drying Machine|washing machine|washing machines
Kettle|kettle|kettles
Tea pot|teapot|teapots
Head Phone|pair of headphones|-
Pie|pie|pies
Soap|!soap|-
Umbrella|umbrella|umbrellas
Lantern|lantern|lanterns
Clock|clock|clocks
Green Onion|green onion|-
Scale|scale|scales
Scissors|pair of scissors|-
SUV|SUV|SUVs
Egg|egg|eggs
Necklace|necklace|necklaces
Pear|pear|pears
Showerhead|showerhead|showerheads
Airplane|airplane|airplanes
Fan|fan|fans
Shovel|shovel|shovels
Wild Bird|bird|birds
Sandals|sandal|-
Lettuce|!lettuce|-
Printer|printer|printers
Stuffed Toy|stuffed toy|stuffed toys
Helmet|helmet|helmets
Pineapple|pineapple|pineapples
Camera|camera|cameras
Watermelon|watermelon|-
Coffee Machine|coffee machine|coffee machines
Induction Cooker|induction cooker|-
Tong|pair of tongs|-
Tie|tie|ties
Jug|jug|jugs
Marker|marker|markers
Crane|construction crane|construction cranes
Gloves|glove|-
Sausage|sausage|sausages
Piano|piano|pianos
Trolley|trolley|trolleys
Leather Shoes|leather shoe|-
Cookies|cookie|cookies
Traffic Light|traffic light|traffic lights
Flag|flag|flags
Belt|belt|belts
Chopsticks|pair of chopsticks|-
Garlic|!garlic|-
Hamburger|hamburger|hamburgers
Cake|cake|cakes
Dog|dog|dogs
Blender|blender|blenders
Peach|peach|peaches
Backpack|backpack|backpacks
Cat|cat|cats
Cello|cello|cellos
earphone|earphone|-
Eggplant|eggplant|eggplants
Motorcycle|motorcycle|motorcycles
Awning|awning|awnings
Tape|roll of tape|-
Traffic cone|traffic cone|traffic cones
Router/modem|router|routers
Mango|mango|mangoes
Ship|ship|ships
Air Conditioner|air conditioner|air conditioners
Toaster|toaster|toasters
Brush|brush|brushes
Lifesaver|life buoy|life buoys
Pizza|pizza|pizzas
Bus|bus|buses
Radiator|radiator|radiators
Nightstand|nightstand|nightstands
Hamimelon|cantaloupe|-
Corn|!corn|-
Pasta|!pasta|-
Kiwi fruit|kiwi|kiwis
Pickup Truck|pickup truck|pickup trucks
Dishwasher|dishwasher|dishwashers
Avocado|avocado|avocados
Candle|candle|candles
Horse|horse|horses
Scooter|scooter|scooters
Saxophone|saxophone|saxophones
Tablet|tablet|tablets
Plum|plum|plums
Mushroon|mushroom|mushrooms
High Heels|high-heeled shoe|-
Tent|tent|tents
Key|key|keys
Wallet/Purse|wallet|wallets
Slide|slide|slides
Boots|boot|-
Train|train|trains
CD|CD|CDs
Ladder|ladder|ladders
Cue|cue stick|cue sticks
Sandwich|sandwich|sandwiches
Goose|goose|geese
Elephant|elephant|elephants
Surveillance Camera|surveillance camera|surveillance cameras
Swing|swing|swings
Fire Extinguisher|fire extinguisher|fire extinguishers
Traffic Sign|traffic sign|traffic signs
Cheese|!cheese|-
Pigeon|pigeon|pigeons
Donut|donut|donuts
Deer|deer|deer
Shrimp|shrimp|-
Projector|projector|projectors
Folder|folder|folders
Steak|steak|-
Broom|broom|brooms
Tape Measur/ Ruler|ruler|rulers
Screwdriver|screwdriver|screwdrivers
Candy|!candy|-
Speed Limit Sign|speed limit sign|speed limit signs
Ambulance|ambulance|ambulances
Pliers|pair of pliers|-
Zebra|zebra|zebras
Cherry|cherry|cherries
Stapler|stapler|staplers
Swan|swan|swans
Hair Dryer|hair dryer|hair dryers
Sailboat|sailboat|sailboats
Stop Sign|stop sign|stop signs
Sports Car|sports car|sports cars
Hot dog|hot dog|hot dogs
Trumpet|trumpet|trumpets
Luggage|suitcase|suitcases
Ice cream|!ice cream|-
Carriage|carriage|carriages
Board Eraser|board eraser|board erasers
Violin|violin|violins
Hammer|hammer|hammers
Pomegranate|pomegranate|pomegranates
Recorder|recorder|recorders
Papaya|papaya|papayas
Calculator|calculator|calculators
Ballon|balloon|balloons
Paint Brush|paintbrush|paintbrushes
Radish|radish|radishes
Heavy Truck|heavy truck|heavy trucks
Giraffe|giraffe|giraffes
Red Cabbage|red cabbage|-
Side Table|side table|side tables
Penguin|penguin|penguins
Coconut|coconut|coconuts
Parking meter|parking meter|parking meters
Coffee Table|coffee table|coffee tables
Rice Cooker|rice cooker|rice cookers
Asparagus|!asparagus|-
Tricycle|tricycle|tricycles
Mask|face mask|face masks
Bow Tie|bow tie|bow ties
Sushi|!sushi|-
Goldfish|goldfish|-
Hotair ballon|hot air balloon|hot air balloons
Megaphone|megaphone|megaphones
Trombone|trombone|trombones
Cow|cow|cows
Pig|pig|pigs
Soccer|soccer ball|soccer balls
Rickshaw|rickshaw|rickshaws
Flute|flute|flutes
Trophy|trophy|trophies
Sheep|sheep|sheep
Golf Club|golf club|golf clubs
Surfboard|surfboard|surfboards
Helicopter|helicopter|helicopters
Golf Ball|golf ball|golf balls
Briefcase|briefcase|briefcases
Grapefruit|grapefruit|grapefruits
Gun|gun|guns
Fire Hydrant|fire hydrant|fire hydrants
Flask|flask|flasks
Mop|mop|mops
Meat ball|meatball|meatballs
Crab|crab|crabs
Fire Truck|fire truck|fire trucks
Fishing Rod|fishing rod|fishing rods
Stroller|stroller|strollers
Comb|comb|combs
Tennis|tennis ball|tennis balls
Hockey Stick|hockey stick|hockey sticks
Paddle|paddle|paddles
Cigar/Cigarette|cigarette|cigarettes
Seal|seal|seals
Kite|kite|kites
Electric Drill|electric drill|electric drills
Urinal|urinal|urinals
Antelope|antelope|antelopes
Donkey|donkey|donkeys
Lighter|lighter|lighters
Eraser|eraser|erasers
Pencil Case|pencil case|pencil cases
Baseball|baseball|baseballs
Noddles|!noodles|-
Tennis Racket|tennis racket|tennis rackets
Globe|globe|globes
Spring Rolls|spring roll|spring rolls
Okra|!okra|-
Campel|camel|camels
Basketball|basketball|basketballs
Dumbbell|dumbbell|dumbbells
Medal|medal|medals
Crosswalk Sign|crosswalk sign|crosswalk signs
Frisbee|frisbee|frisbees
Parrot|parrot|parrots
Lion|lion|lions
Dolphin|dolphin|dolphins
Hurdle|hurdle|hurdles
Wheelchair|wheelchair|wheelchairs
Treadmill|treadmill|treadmills
Rabbit|rabbit|rabbits
Scallop|scallop|scallops
Poker Card|playing card|playing cards
Cosmetics Brush/Eyeliner Pencil|makeup brush|-
Baseball Bat|baseball bat|baseball bats
Lipstick|lipstick|lipsticks
Egg tart|egg tart|egg tarts
Tuba|tuba|tubas
Buttefly|butterfly|butterflies
Jellyfish|jellyfish|-
Bear|bear|bears
Lobster|lobster|lobsters
American Football|football|footballs
Monkey|monkey|monkeys
Baozi|steamed bun|steamed buns
Volleyball|volleyball|volleyballs
Target|target|targets
Cosmetics Mirror|makeup mirror|makeup mirrors
Skateboard|skateboard|skateboards
Dumpling|dumpling|dumplings
Durian|durian|durians
Formula 1|race car|race cars
Yak|yak|yaks
Oyster|oyster|oysters
Snowboard|snowboard|snowboards
Baseball Glove|baseball glove|baseball gloves
Skiboard|ski|skis
French|French horn|French horns
Chainsaw|chainsaw|chainsaws
Binoculars|pair of binoculars|-
Barbell|barbell|barbells
Game board|game board|game boards
Hoverboard|hoverboard|hoverboards
Table Teniis paddle|table tennis paddle|table tennis paddles
Table Tennis|ping pong ball|ping pong balls
Curling|curling stone|curling stones
"""
# Never asked: vague or ambiguous labels.
EXCLUDED = {"Other Shoes", "Other Fish", "Other Balls", "Toiletry", "Cleaning Products", "Converter", "Canned",
            "Green Vegetables", "Dessert", "Cosmetics", "Nuts", "Billards", "Chips", "Notepaper", "Grape",
            "Green beans", "French Fries", "Chicken", "Skating and Skiing shoes"}

# Classes a person could call by each other's name: never a negative / count-0 next to one another, never a pair.
CONFUSION = [
    "Car|SUV|Van|Pickup Truck|Sports Car|Truck|Heavy Truck|Machinery Vehicle|Fire Truck|Ambulance|Bus|Formula 1|"
    "Train|Carriage|Rickshaw|Tricycle|Trolley",
    "Bicycle|Motorcycle|Scooter|Tricycle|Hoverboard|Rickshaw",
    "Boat|Ship|Sailboat", "Airplane|Helicopter|Hotair ballon|Ballon",
    "Sneakers|Other Shoes|Leather Shoes|Boots|High Heels|Slippers|Sandals|Skating and Skiing shoes",
    "Cup|Wine Glass|Jug|Kettle|Tea pot|Flask|Bottle|Canned|Vase",
    "Desk|Dinning Table|Side Table|Coffee Table|Nightstand|Cabinet/shelf",
    "Chair|Stool|Couch|Bench|Wheelchair|Swing",
    "Handbag/Satchel|Backpack|Luggage|Briefcase|Wallet/Purse|Pencil Case|Storage box",
    "Wild Bird|Duck|Goose|Pigeon|Swan|Parrot|Penguin|Chicken",
    "Other Fish|Goldfish|Shrimp|Lobster|Crab|Scallop|Oyster|Dolphin|Seal|Jellyfish",
    "Orange/Tangerine|Lemon|Grapefruit|Mango|Peach|Apple|Pear|Plum|Pomegranate|Papaya",
    "Watermelon|Hamimelon|Papaya|Durian|Coconut",
    "Green Vegetables|Cabbage|Lettuce|Red Cabbage|Broccoli|Green Onion|Asparagus|Okra|Green beans",
    "Pepper|Tomato|Eggplant|Cucumber|Radish|Carrot|Potato|Onion|Garlic",
    "Moniter/TV|Laptop|Tablet|Computer Box|Projector|Blackboard/Whiteboard",
    "Cell Phone|Telephone|Calculator|Remote|Tablet",
    "Lamp|Street Lights|Lantern|Candle",
    "Storage box|Barrel/bucket|Bakset|Trash bin Can|Canned|Pot|Vase|Bowl/Basin",
    "Bowl/Basin|Plate|Pot|Sink|Bathtub",
    "Gas stove|Induction Cooker|Oven|Microwave|Rice Cooker|Toaster|Extractor",
    "Pen/Pencil|Marker|Cosmetics Brush/Eyeliner Pencil|Paint Brush|Brush|Lipstick|Eraser",
    "Traffic Sign|Speed Limit Sign|Stop Sign|Crosswalk Sign|Traffic Light|Flag",
    "Other Balls|Soccer|Basketball|Volleyball|Baseball|Golf Ball|Tennis|American Football|Table Tennis|Ballon",
    "Tennis Racket|Table Teniis paddle|Paddle|Baseball Bat|Golf Club|Hockey Stick|Cue",
    "Dessert|Cake|Pie|Cookies|Egg tart|Ice cream|Donut|Candy|Bread|Baozi|Dumpling|Hamburger|Sandwich|Pizza",
    "Mirror|Cosmetics Mirror|Picture/Frame",
    "Cow|Yak|Antelope|Deer|Sheep|Donkey|Horse|Campel|Pig",
    "Head Phone|earphone|Microphone",
    "Bracelet|Ring|Necklace|Watch",
    "Glasses|Binoculars|Mask",
    "Cello|Violin|Guitar", "Trumpet|Trombone|Tuba|French|Saxophone|Flute|Recorder|Megaphone",
    "Drum|Cymbal",
    "Skiboard|Snowboard|Surfboard|Skateboard|Hoverboard",
    "Sink|Bathtub|Urinal|Toilet",
    "Washing Machine/Drying Machine|Dishwasher|Refrigerator",
    "Hat|Helmet", "Tissue|Napkin|Toilet Paper|Towel",
    "Dumbbell|Barbell", "Speaker|Router/modem|Computer Box",
    "Surveillance Camera|Camera",
]


def _parse_table() -> Dict[str, Tuple[str, Optional[str]]]:
    out = {}
    for line in _TABLE.strip().splitlines():
        raw, sing, plur = (x.strip() for x in line.split("|"))
        out[raw] = (sing, None if plur == "-" else plur)
    return out


NAMES = _parse_table()
GROUPS: Dict[str, Set[int]] = defaultdict(set)
for _i, _g in enumerate(CONFUSION):
    for _c in _g.split("|"):
        GROUPS[_c.strip()].add(_i)
_GENERIC = {"or", "of", "pair", "roll", "and", "the", "machine", "table", "box"}


def norm(label: str) -> str:
    return label.strip()


def asked(label: str) -> bool:
    return norm(label) in NAMES and norm(label) not in EXCLUDED


@lru_cache(maxsize=None)
def head_words(label: str) -> FrozenSet[str]:
    sing, plur = NAMES.get(norm(label), (norm(label).lower(), None))
    words = re.findall(r"[a-z]+", (sing + " " + (plur or "")).lower())
    return frozenset(w.rstrip("s") for w in words if len(w) >= 3 and w not in _GENERIC)


@lru_cache(maxsize=None)
def confusable(a: str, b: str) -> bool:
    """True when `a` and `b` could be called by each other's name (same confusion group or a shared word,
    e.g. "traffic sign" / "stop sign", "cabbage" / "red cabbage")."""
    a, b = norm(a), norm(b)
    if a == b or GROUPS.get(a, set()) & GROUPS.get(b, set()):
        return True
    return bool(head_words(a) & head_words(b))


def thing(label: str) -> str:
    """'a basket' / 'an apple' / 'an SUV' / 'rice' (mass nouns have no article)."""
    sing = NAMES[norm(label)][0]
    if sing.startswith("!"):
        return sing[1:]
    an = sing[0].lower() in "aeiou" or sing in ("SUV",) or sing.startswith("hour")
    if sing.lower().startswith(("uni", "use", "eu")):
        an = False
    return ("an " if an else "a ") + sing


# ---------------------------------------------------------------- facts of one picture (pure)

def boxes_by_class(anns: Sequence[dict], width: float, height: float) -> Dict[str, List[dict]]:
    """Per raw class label, its boxes as {area: share of the picture, cx: centre x / width, crowd, real}."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for a in anns:
        x0, y0, x1, y1 = a["bbox"]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / max(1.0, width * height)
        out[norm(a["category"])].append({"area": area, "cx": (x0 + x1) / 2 / max(1.0, width),
                                         "crowd": bool(a.get("iscrowd")),
                                         "real": not a.get("isfake") and not a.get("isreflected")})
    return out


def visible(boxes: Dict[str, List[dict]]) -> List[str]:
    """Classes (asked ones) whose largest real box covers >= MIN_AREA: safe "yes" answers."""
    return sorted(c for c, bs in boxes.items()
                  if asked(c) and max((b["area"] for b in bs if b["real"]), default=0) >= MIN_AREA)


def count_of(boxes: Dict[str, List[dict]], label: str) -> Optional[int]:
    """The number of `label` in the picture, or None when it cannot be known or asked: a crowd box, a reflection
    / fake, a box under MIN_AREA, more than MAX_COUNT, or a class that is not counted. 0 when absent."""
    label = norm(label)
    if not asked(label) or NAMES[label][1] is None:
        return None
    bs = boxes.get(label, [])
    if any(b["crowd"] or not b["real"] or b["area"] < MIN_AREA for b in bs) or len(bs) > MAX_COUNT:
        return None
    return len(bs)


def absent_classes(boxes: Dict[str, List[dict]], cooc: Dict[str, Counter],
                   countable: bool = False) -> Tuple[List[str], List[int]]:
    """Asked classes absent from the picture (exhaustive annotation: no box of it at all) that co-occur with the
    picture's classes, with weight = how often; never confusable with a class in the picture."""
    present = set(boxes)
    weight: Counter = Counter()
    for p in present:
        for c, n in cooc.get(p, {}).items():
            weight[c] += n
    cands = sorted(c for c in weight if c not in present and asked(c)
                   and (not countable or NAMES[c][1] is not None)
                   and not any(confusable(c, p) for p in present))
    return cands, [weight[c] for c in cands]


def plausible_absent(boxes: Dict[str, List[dict]], cooc: Dict[str, Counter], rng: random.Random,
                     countable: bool = False, k: int = 1) -> List[str]:
    """Up to `k` distinct absent classes (absent_classes), drawn by co-occurrence weight."""
    cands, weights = absent_classes(boxes, cooc, countable)
    out: List[str] = []
    while cands and len(out) < k:
        c = rng.choices(cands, weights=weights)[0]
        i = cands.index(c)
        cands.pop(i)
        weights.pop(i)
        out.append(c)
    return out


def left_right_pairs(boxes: Dict[str, List[dict]]) -> List[Tuple[str, str, bool]]:
    """(a, b, a_is_left) for classes with exactly one box each (real, not crowd, >= MIN_AREA) whose centres are
    more than MIN_GAP of the width apart, a < b by name, never two confusable classes."""
    single = sorted(c for c, bs in boxes.items() if asked(c) and len(bs) == 1 and not bs[0]["crowd"]
                    and bs[0]["real"] and bs[0]["area"] >= MIN_AREA)
    out = []
    for i, a in enumerate(single):
        for b in single[i + 1:]:
            ca, cb = boxes[a][0]["cx"], boxes[b][0]["cx"]
            if abs(ca - cb) > MIN_GAP and not confusable(a, b):
                out.append((a, b, ca < cb))
    return out


def cooccurrence(pictures: Iterable[Iterable[str]]) -> Dict[str, Counter]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for cats in pictures:
        cs = sorted({norm(c) for c in cats})
        for a in cs:
            for b in cs:
                if a != b:
                    out[a][b] += 1
    return out


# ---------------------------------------------------------------- questions

EXIST = ["Is there {t} in the image?", "Is there {t} in this picture?", "Can you see {t}?",
         "Does this image contain {t}?", "Is there {t} in the photo?"]
COUNT = ["How many {p} are there?", "How many {p} are in the image?", "How many {p} can you see?",
         "How many {p} are in this picture?"]
POSITION = ["Is the {a} to the left or to the right of the {b}?", "Is the {a} on the left or the right of the {b}?",
            "Relative to the {b}, is the {a} on the left or on the right?"]


def exist_question(sid, image_id, image, label, yes, rng):
    q = rng.choice(EXIST).format(t=thing(label))
    return noul(sid, SOURCE, "exist", image_id, image, q, yes, group="objects365-exist-%s" % norm(label))


def count_question(sid, image_id, image, label, n, rng):
    q = rng.choice(COUNT).format(p=NAMES[norm(label)][1])
    return choice(sid, SOURCE, "count", image_id, image, q, count_options(n, rng), str(n), rng,
                  group="objects365-count", label=norm(label))


def position_question(sid, image_id, image, a, b, a_is_left, rng):
    """'Is the a to the left or right of the b?' (options left / right). `pair` (the two labels, sorted) and
    `subject` (a) key the balance and are removed before writing."""
    q = rng.choice(POSITION).format(a=NAMES[norm(a)][0].lstrip("!"), b=NAMES[norm(b)][0].lstrip("!"))
    return choice(sid, SOURCE, "position", image_id, image, q, ["left", "right"], "left" if a_is_left else "right",
                  rng, group="objects365-position", pair="|".join(sorted((norm(a), norm(b)))), subject=norm(a))


def _key(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", norm(label)).strip("_").lower()


def questions_for(pid: str, image: str, boxes: Dict[str, List[dict]], cooc: Dict[str, Counter],
                  rng: random.Random, per_kind: int = 2) -> List[dict]:
    """The candidate questions of one picture, up to `per_kind` of each: exist yes (visible classes), exist no
    (plausible absent classes), count (a countable class; now and then 0 for a plausible absent class), left /
    right (pairs of single-instance classes). The balancing and the per-picture cap choose among them."""
    image_id = "objects365:%s" % pid
    sid = "ext-objects365:%s:%%s:%%s" % pid
    out = []
    vis = visible(boxes)
    for c in rng.sample(vis, min(per_kind, len(vis))):
        out.append(exist_question(sid % ("exist", _key(c)), image_id, image, c, True, rng))
    for c in plausible_absent(boxes, cooc, rng, k=per_kind):
        out.append(exist_question(sid % ("exist", _key(c)), image_id, image, c, False, rng))
    countable = [c for c in sorted(boxes) if (count_of(boxes, c) or 0) > 0]
    for c in rng.sample(countable, min(per_kind, len(countable))):
        out.append(count_question(sid % ("count", _key(c)), image_id, image, c, count_of(boxes, c) or 0, rng))
    if rng.random() < 0.3 or not countable:
        for c in plausible_absent(boxes, cooc, rng, countable=True):
            out.append(count_question(sid % ("count", _key(c)), image_id, image, c, 0, rng))
    pairs = left_right_pairs(boxes)
    for a, b, a_left in rng.sample(pairs, min(per_kind, len(pairs))):
        if rng.random() < 0.5:
            a, b, a_left = b, a, not a_left
        out.append(position_question(sid % ("position", _key(a) + "-" + _key(b)), image_id, image, a, b, a_left,
                                     rng))
    return out


def balance_positions(rows: List[dict]) -> List[dict]:
    """Per unordered class pair, as many questions where the alphabetically first class is on the left as on the
    right, so neither the pair nor the wording gives the answer away; other rows untouched."""
    by: Dict[str, Dict[bool, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] == "position":
            first_is_subject = r["pair"].split("|")[0] == r["subject"]
            by[r["pair"]][first_is_subject == (r["answer_key"] == "left")].append(r)
    keep = set()
    for sides in by.values():
        k = min(len(sides[True]), len(sides[False]))
        for v in sides.values():
            keep |= {id(r) for r in v[:k]}
    return [r for r in rows if r["kind"] != "position" or id(r) in keep]


def flatten_counts(rows: List[dict]) -> List[dict]:
    """Count rows: per class, no answer more frequent than the class's second answer (so "how many people" alone
    says little), then over all classes no answer above twice the mean (common.flatten_answers)."""
    from v2_common import flatten_top  # pyright: ignore[reportMissingImports]
    counts = [dict(r, group=r["label"]) for r in rows if r["kind"] == "count"]
    counts = [dict(r, group="objects365-count") for r in flatten_answers(flatten_top(counts))]
    return [r for r in rows if r["kind"] != "count"] + counts


def cap_mixed(rows: List[dict], cap: int, rng: random.Random) -> List[dict]:
    """At most `cap` questions per picture, one of each kind before a second of any kind."""
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["image_id"]].append(r)
    out = []
    for k in sorted(by):
        v = by[k]
        rng.shuffle(v)
        seen: Counter = Counter()
        ranked = []
        for r in v:
            ranked.append((seen[r["kind"]], len(ranked), r))
            seen[r["kind"]] += 1
        out += [r for _, _, r in sorted(ranked, key=lambda t: t[:2])[:cap]]
    return out


def balance(rows: List[dict], per_picture: int, rng: random.Random) -> Tuple[List[dict], Dict[str, int]]:
    """yes = no per class, left / right per pair, counts flattened; then the per-picture cap and the yes / no and
    left / right balance once more (the cap removes rows unevenly)."""
    steps = {}
    rows = flatten_counts(balance_positions(balance_yes_no(rows)))
    steps["after_balance"] = len(rows)
    rows = cap_mixed(rows, per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = balance_positions(balance_yes_no(rows))
    steps["after_rebalance"] = len(rows)
    return rows, steps


# ---------------------------------------------------------------- data

def annotation_rows(root: str, patches: Sequence[int]):
    """(file name, id, width, height, anns) of every training picture in `patches`, from all annotation shards."""
    import pyarrow.parquet as pq
    wanted = {"patch%d" % p for p in patches}
    for path in hf_files(root, ANN_REPO, "data/train-*.parquet"):
        t = pq.read_table(path, columns=["image_path", "image_info", "anns_info"])
        keep = [i for i, x in enumerate(t.column("image_path").to_pylist()) if x.split("/")[2] in wanted]
        if not keep:
            continue
        t = t.take(keep)
        for path_, info, anns in zip(t.column("image_path").to_pylist(), t.column("image_info").to_pylist(),
                                     t.column("anns_info").to_pylist()):
            yield os.path.basename(path_), info["id"], info["width"], info["height"], anns or []


def extract(root: str, patches: Sequence[int], wanted: Dict[str, str]) -> Tuple[int, int]:
    """Save the pictures `wanted` (file name -> rel path) from the patch archives; existing files are skipped and
    an archive is not opened when all its pictures are there. Returns (extracted now, missing)."""
    todo = {n: r for n, r in wanted.items() if not os.path.exists(os.path.join(root, r))}
    got = 0
    for p in patches:
        if not todo:
            break
        with tarfile.open(hf_file(root, IMG_REPO, "patch%d.tar.gz" % p), "r|gz") as tar:
            for m in tar:
                name = os.path.basename(m.name)
                if m.isfile() and name in todo:
                    f = tar.extractfile(m)
                    if f:
                        save_bytes(os.path.join(root, todo.pop(name)), f.read())
                        got += 1
                        if got % 5000 == 0:
                            print("  extracted %d pictures" % got, flush=True)
    return got, len(todo)


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--images", type=int, default=54000, help="pictures to use (before held-out drops)")
    p.add_argument("--patches", default="0,1", help="training archives patch<k>.tar.gz to take pictures from")
    p.add_argument("--per-picture", type=int, default=3)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    patches = [int(x) for x in a.patches.split(",")]
    rng = random.Random(a.seed)
    steps: Counter = Counter()
    pool = list(annotation_rows(a.root, patches))
    steps["pictures_in_patches"] = len(pool)
    cooc = cooccurrence([x["category"] for x in anns] for _, _, _, _, anns in pool)
    pool.sort(key=lambda x: hashlib.sha1(x[0].encode()).hexdigest())     # deterministic, patch-independent order
    rows, wanted = [], {}
    for name, pid, w, h, anns in pool:
        if len(wanted) >= a.images:
            break
        rel = os.path.join("ext", SOURCE, "images", name)
        qs = questions_for(str(pid), rel, boxes_by_class(anns, w, h), cooc, rng)
        if qs:
            wanted[name] = rel
            rows += qs
    steps["pictures_selected"] = len(wanted)
    steps["questions_candidates"] = len(rows)
    got, missing = extract(a.root, patches, wanted)
    steps["pictures_extracted_now"], steps["pictures_missing_in_archives"] = got, missing
    rows = [r for r in rows if os.path.exists(os.path.join(a.root, r["image"]))]
    near = HeldOut(a.root).near_held({r["image"] for r in rows}, SOURCE)
    steps["pictures_near_held_out"] = len(near)
    rows = [r for r in rows if r["image"] not in near]
    steps["questions_after_held_out"] = len(rows)
    rows, bsteps = balance(rows, a.per_picture, rng)
    steps.update(bsteps)
    for r in rows:
        for k in ("pair", "subject", "label"):
            r.pop(k, None)
    finish(a.root, SOURCE, rows, dict(steps), {
        "licence": "annotations CC BY 4.0, academic use only; pictures may not be redistributed",
        "route": "raw Objects365 v2 train annotations (%s, parquet) + picture archives %s from %s"
                 % (ANN_REPO, ", ".join("patch%d.tar.gz" % p for p in patches), IMG_REPO),
        "downloads_bytes": {os.path.basename(f): os.path.getsize(f) for f in
                            hf_files(a.root, ANN_REPO, "data/train-*.parquet")
                            + [hf_file(a.root, IMG_REPO, "patch%d.tar.gz" % p) for p in patches]},
        "position_answers": dict(Counter(r["answer_key"] for r in rows if r["kind"] == "position")),
        "count_answers": dict(sorted(Counter(r["answer_key"] for r in rows if r["kind"] == "count").items(),
                                     key=lambda kv: int(kv[0]))),
        "exist_classes": len({r["group"] for r in rows if r["kind"] == "exist"})})


if __name__ == "__main__":
    main()
