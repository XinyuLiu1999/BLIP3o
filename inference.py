from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import pdb
import copy
import sys
import argparse
import os
import json
from tqdm import tqdm
import shortuuid
from blip3u.constants import *
from blip3u.conversation import conv_templates, SeparatorStyle
from blip3u.model.builder import load_pretrained_model
from blip3u.utils import disable_torch_init
from blip3u.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import requests
from blip3u.conversation import conv_templates, SeparatorStyle
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info

import re, random

diffusion_path = "/fsx/home/jiuhai.chen/hub/models--BAAI--Emu2-Gen/snapshots/a41a2dcd777a68225dddc72c7213b064ee06f4a0"

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


# QwenVL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map={"": 1}
# )





model_path = sys.argv[1]
save_folder = sys.argv[2]

device_1 = 0
device_2 = 1


disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)




pipe = DiffusionPipeline.from_pretrained(
   diffusion_path,
   custom_pipeline="pipeline_blip3u_gen",
   torch_dtype=torch.bfloat16,
   use_safetensors=True,
   variant="bf16",
   multimodal_encoder=multi_model,
   tokenizer=tokenizer,
   safety_checker=None
)


pipe.vae.to(f'cuda:{device_1}')
pipe.unet.to(f'cuda:{device_1}')
# QwenVL.to(f'cuda:{device_2}')





captions_1 = [
    "Photo of a bear wearing a suit and tophat in a river in the middle of a forest holding a sign that says 'I can't bear it'.",
    "A crab made of cheese on a plate.",
    "Dystopia of thousands of workers picking cherries and feeding them into a machine that runs on steam and is as large as a skyscraper. Written on the side of the machine: 'Transfusion'.",
    "A car made out of vegetables.",
    "Detailed pen and ink drawing of a happy pig butcher selling meat in its shop.",
    "A massive alien spaceship that is shaped like a pretzel.",
    "A kangaroo holding a beer, wearing ski goggles and passionately singing silly songs.",
    "Translucent pig, inside is a smaller pig.",
    "Film still of a long-legged cute big-eye anthropomorphic cheeseburger wearing sneakers relaxing on the couch in a sparsely decorated living room.",
    "Human life depicted entirely out of fractals.",
    "This dreamlike digital art captures a vibrant, kaleidoscopic bird in a lush rainforest.",
    "A small office made out of car parts.",
    "A space elevator, cinematic sci-fi art.",
    "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus.",
    "Smiling cartoon dog sits at a table, coffee mug in hand, as a room goes up in flames. 'This is fine,' the dog assures himself.",
    "An old rusted robot wearing pants and a jacket riding skis in a supermarket.",
    "Dark high contrast render of a psychedelic tree of life illuminating dust in a mystical cave.",
    "Tilt-shift aerial photo of a cute city made of sushi on a wooden table in the evening.",
    "Beautiful oil painting of a steamboat in a river in the afternoon. On the side of the river is a large brick building with a sign on top that says 'Transfusion'.",
    "'GO BIG OR GO UNET' is written on the blackboard.",
    "A wine glass on top of a dog.",
    "A photo of a confused grizzly bear in calculus class.",
    "A brown bird and a blue bear.",
    "A small blue book sitting on a large red book.",
    "A pear cut into seven pieces arranged in a ring.",
    "A wall in a royal castle. There are two paintings on the wall. The one on the left is a detailed oil painting of the royal raccoon king. The one on the right is a detailed oil painting of the royal raccoon queen.",
    "A chrome-plated duck with a golden beak arguing with an angry turtle in a forest.",
    "A cloud in the shape of two bunnies playing with a ball. The ball is made of clouds too.",
    "A DSLR picture of colorful graffiti showing a hamster with a moustache.",
    "An angry duck doing heavy weightlifting at the gym.",
    "A photo of a person with the head of a cow, wearing a tuxedo and black bowtie. Beach wallpaper in the background.",
    "A family of three houses in a meadow. The Dad house is a large blue house. The Mom house is a large pink house. The Child house is a small wooden shed.",
    "Android Mascot made from bamboo.",
    "A chromeplated cat sculpture placed on a Persian rug.",
    "Three spheres made of glass falling into the ocean. Water is splashing. Sun is setting.",
    "A transparent sculpture of a duck made out of glass.",
    "A raccoon wearing a cowboy hat and black leather jacket is behind the backyard window. Rain droplets on the window.",
    "A bucket bag made of blue suede. The bag is decorated with intricate golden paisley patterns. The handle of the bag is made of rubies and pearls.",
    "Intricate origami of a fox and a unicorn in a snowy forest.",
    "A relaxed garlic with a blindfold reading a newspaper while floating in a pool of tomato soup.",
    "A photo of a corgi dog wearing a wizard hat playing guitar on the top of a mountain.",
    "A single beam of light enters the room from the ceiling. The beam of light is illuminating an easel. On the easel, there is a Rembrandt painting of a raccoon.",
    "A blue jay standing on a large basket of rainbow macarons.",
    "The Toronto skyline with Meta AI logo written in fireworks.",
    "An art gallery displaying Monet paintings. The art gallery is flooded. Robots are going around the art gallery using paddle boards.",
    "An armchair in the shape of an avocado.",
    "An expressive oil painting of a chocolate chip cookie being dipped in a glass of milk, depicted as an explosion of flavors.",
    "An illustration of an avocado sitting in a therapist's chair, saying 'I just feel so empty inside' with a pit-sized hole in its center. The therapist, a spoon, scribbles notes.",
    "A Dutch still life of an arrangement of tulips in a fluted vase. The lighting is subtle, casting gentle highlights on the flowers and emphasizing their delicate details and natural beauty.",
    "A Baroque-style painting depicting a cat on a wooden rocking chair napping in the sunlight.",
    "A tranquil, anime-style koi pond in a serene Japanese garden, featuring blossoming cherry trees.",
    "Photo of a lychee-inspired spherical chair, with a bumpy white exterior and plush interior, set against a tropical wallpaper.",
    "A spacious, serene room influenced by modern Japanese aesthetics with a view of a cityscape outside of the window.",
    "Against a black backdrop, a middle-aged Tongan woman twirls, her skin glowing and curly hair flowing. She wears an outfit resembling a whirlwind of marble and porcelain, illuminated by shard gleams, creating a dreamlike, fragmented yet fluid appearance.",
    "A vibrant painting of a landscape with a dynamic blue sky filled with fluffy cumulus clouds and streaks of sunlight. Below, green grass sways in the wind, highlighted by the sun's rays.",
    "A 2D animation of a folk music band composed of anthropomorphic autumn leaves, each playing traditional bluegrass instruments, amidst a rustic forest setting dappled with the soft light of a harvest moon.",
    "White Cycladic houses with blue accents and vibrant magenta bougainvillea in a serene Greek island setting.",
    "Turtle swimming underwater, aesthetic, fantasy.",
    "Elephant swimming underwater, aesthetic, fantasy.",
    "Flock of sheep, aesthetic, fantasy.",
    "Open hand, hand model. 4k. White background.",
    "Fist, hand model. 4k. White background.",
    "Chameleon and octopus, side by side, high-quality render, drawing, professional.",
    "A plush toy koala bear relaxing on a lounge chair and working on a laptop. The chair is beside a rose flower pot. There is a window on the wall beside the flower pot with a view of snowy mountains.",
    "A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.",
    "A teddy bear wearing a motorcycle helmet and cape is riding a motorcycle in Rio de Janeiro with Dois Irmãos in the background. DSLR photo.",
    "A black German shepherd wearing a red beret.",
    "An Armenian church on the surface of Mars, with an astronaut walking into the church, in focus. Photo. Fantasy. Dramatic.",
    "Armenian khachkars surrounded by pomegranates in a bright green forest.",
    "A cat wearing sunglasses.",
    "A small cactus wearing a straw hat and neon sunglasses in the Sahara desert.",
    "A close-up photo of a human hand, hand model. High quality.",
    "A raccoon main character in an anime preparing for an epic battle with a samurai sword. Battle stance. Fantasy, Illustration.",
    "Stop sign in a fantasy style with the text '1992 Spring'.",
    "A cute bear holds a stop sign that says 'Transfusion'.",
    "A sign that says 'Transfusion'.",
    "A rowboat on a lake with a bike on it.",
    "A glass of orange juice.",
    "A woman in an apron works at a local bar.",
    "A coffee mug.",
    "An egg and a bird made of wheat bread.",
    "A corgi.",
    "A shake is next to a cake.",
    "An elephant walking out of a fridge.",
    "A monarch butterfly.",
    "A beaver dressed in a vest, wearing glasses and a vibrant necktie, in a library.",
    "An emu wearing sunglasses and chilling on a beach.",
    "A bread, an apple, and a knife on a table.",
    "A heart made of wood.",
    "An old man with green eyes and a long grey beard.",
    "A painting of an adorable rabbit sitting on a colorful splash.",
    "The oil painting shows a cow standing near a tree with red leaves.",
    "A traditional tea house in a tranquil garden with blooming cherry blossom trees.",
    "A painting of trees near a peaceful lake.",
    "An afrofuturist lady wearing gold jewelry.",
    "A black basketball shoe with a lightning bolt on it.",
    "A cool orange cat wearing sunglasses playing a guitar with a group of dancing bananas.",
    "A horse reading a book.",
    "Eerie man, but not genuinely frightening.",
    "A woman on a bed underneath a blanket.",
    "A man is meditating on a beach at sunrise, 4k.",
    "A light bulb containing a sailboat floats through the galaxy.",
    "A group of toucans on an Athenian vase, painted in Egyptian style.",
    "A wooden deck.",
    "Sunset in a valley with trees and mountains.",
    "A sign that says 'Diffusion'.",
    "A 1960s yearbook photo with animals dressed as humans.",
    "A stack of 3 cubes. A red cube is on the top, sitting on a red cube. The red cube is in the middle, sitting on a green cube. The green cube is on the bottom.",
    "An emoji of a baby panda wearing a red hat, green gloves, red shirt, and green pants.",
    "The word 'START' on a blue t-shirt.",
    "A logo of a wombat on a coffee mug.",
    "The Mona Lisa.",
    "Graffiti of a funny dog on a street wall.",
    "A photograph of a bust of Homer.",
    "A crayon drawing of a space elevator.",
    "An espresso machine that makes coffee from human souls, high-contrast painting.",
    "A stained glass window depicting a calm Tyrannosaurus rex.",
    "Portrait of a gecko wearing a train conductor's hat and holding a flag that has a yin-yang symbol on it. Woodcut.",
    "Portrait of a gecko wearing a train conductor's hat and holding a flag that has a yin-yang symbol on it. Oil on canvas.",
    "Portrait of a gecko wearing a train conductor's hat and holding a flag that has a yin-yang symbol on it. Watercolor.",
    "A photo of an Athenian vase with a painting of toucans playing tennis in the style of Egyptian hieroglyphics.",
    "A photo of a crocodile made of water.",
    "Downtown Seattle at sunrise. Detailed ink wash.",
    "A yellow wall with two framed sketches.",
    "A robot cooking in the kitchen.",
    "A two-lane road with a bright yellow line.",
    "The saying 'BE EXCELLENT TO EACH OTHER' written in a stained glass window.",
    "The flag of the United Kingdom painted in rusty corrugated iron.",
    "A cloud in the shape of an elephant.",
    "A surrealist dream-like oil painting by Salvador Dalí of a cat playing checkers.",
    "A cute bear holds a stop sign that says 'Mixture of Transformer'.",
    "A sign that says 'Mixture of Transformer'.",
    "Dystopia of thousands of workers picking cherries and feeding them into a machine that runs on steam and is as large as a skyscraper. Written on the side of the machine: 'Mixture of Transformer'.",
    "Beautiful oil painting of a steamboat in a river in the afternoon. On the side of the river is a large brick building with a sign on top that says 'Mixture of Transformer'.",
    "A cute bear holds a stop sign that says 'Transfusion MOT'.",
    "A sign that says 'Transfusion MOT'.",
    "Dystopia of thousands of workers picking cherries and feeding them into a machine that runs on steam and is as large as a skyscraper. Written on the side of the machine: 'Transfusion MOT'.",
    "Beautiful oil painting of a steamboat in a river in the afternoon. On the side of the river is a large brick building with a sign on top that says 'Transfusion MOT'.",
    "'GO BIG OR GO MOT' is written on the blackboard."
    "A serene sunset",
    "A snowy mountain",
    "A blooming rose",
    "A starry night sky",
    "A tranquil beach",
    "A dense rainforest scene with lush green foliage, towering trees with thick vines hanging down, and a waterfall cascading into a clear, serene pool. In the foreground, a brilliantly colored toucan perches on a branch, its vibrant beak contrasting with the deep green leaves. Sunlight filters through the dense canopy, casting a mystical glow.",
    "A peaceful underwater scene featuring a coral reef teeming with life. Brightly colored fish of all shapes and sizes swim through the water, and a sea turtle glides gracefully by. The light from above filters down, casting a beautiful blue glow over the entire scene, illuminating the corals in shades of red, orange, and pink.",
    "A magical forest glade illuminated by bioluminescent flowers and floating fireflies. In the center of the glade, a serene fairy with translucent wings sits on a mushroom, surrounded by soft glowing lights. The night sky above is filled with stars, and the air feels enchanted and full of mystery.",
    "A tranquil lakeside sunrise, soft pastel hues.",
    "A charming village",
    "A luminous fairy garden, bioluminescent flowers.",
    "Gentle snowfall in an ancient forest clearing.",
    "A quiet Venetian canal at twilight.",
    "Iridescent coral reef teeming with life.",
    "A cozy cabin",
    "A cheerful rainbow",
    "A sparkling river",
    "A vast lavender field in Provence, stretching as far as the eye can see under a golden sunset. Rows of vibrant purple flowers create a stunning pattern, and the distant farmhouse with its rustic charm adds to the idyllic scene. Bees buzz gently over the blooms, completing the peaceful landscape.",
    "A coastal cliffside in Ireland, with waves crashing against the rocks far below. The rugged cliffs are covered in green grass and wildflowers, and a lighthouse stands proudly on the edge, its beam cutting through the early morning mist. The sky is overcast, adding to the moody and dramatic atmosphere.",
]




captions = [
    'The image depicts a lively and cheerful office environment where three individuals appear to be celebrating or expressing excitement. The person on the left, wearing a gray blazer and jeans, is mid-jump with one leg raised, holding papers in one hand and throwing another piece of paper into the air. The individual in the middle, dressed in a brown jacket and jeans, is also jumping with one arm raised, holding a piece of paper. The person on the right, wearing a blue shirt and dark pants, is holding a framed picture and appears to be in motion, possibly dancing or moving energetically. The office is modern and well-lit, featuring white walls, a large window allowing natural light, and various plants adding a touch of greenery. The desks are equipped with laptops and other office supplies, indicating a professional setting. The overall atmosphere suggests a moment of joy and celebration among colleagues.',
    "The image depicts a tall, white cylindrical tower standing against a clear blue sky. The tower appears to be a communication or observation structure, as indicated by the antenna at its peak and the presence of what looks like maintenance workers ascending it. The base of the tower is surrounded by a red-brick building with a tiled roof, and there's a small tree visible on the left side of the frame. The overall scene suggests a sunny day with good visibility, emphasizing the height and isolation of the tower in the open sky.",
    "The image captures a serene urban scene at dusk, featuring a prominent obelisk monument at the center of a plaza. The monument is surrounded by a reflective water fountain that mirrors its grandeur and the surrounding trees. The sky is a deep blue, transitioning into a lighter hue near the horizon, with a full moon casting a soft glow over the area. Tall, modern buildings flank the plaza, their windows illuminated, suggesting an active city life beyond the tranquil setting. The trees lining the plaza are lush and green, adding a touch of nature to the urban landscape. The overall atmosphere is calm and inviting, with the interplay of light and shadow enhancing the beauty of the scene.",
    "The image depicts a serene outdoor scene with a woman walking along a paved pathway surrounded by lush greenery and climbing plants. She is carrying a baby in a carrier, both of them smiling warmly at the camera. The pathway is flanked by vibrant green foliage, creating a natural and tranquil atmosphere. The sunlight filters through the leaves, casting a soft glow on the scene, enhancing the peaceful ambiance. The woman is dressed casually, suitable for a leisurely stroll in nature, while the baby appears content and secure in her carrier. The overall mood of the image is one of happiness and tranquility, capturing a moment of simple joy in a beautiful setting.",
    "The national flag of the country where Yellowstone National Park is located.",
    "The animal associated with having (2+7) lives.",
    "A photo of bench.",
    "A golden retriever lying peacefully on a wooden porch, with autumn leaves scattered around.",
    "A young woman with freckles wearing a straw hat, standing in a golden wheat field.",
    "Capture a close-up shot of a vibrant sunflower in full bloom, with a honeybee perched on its petals, its delicate wings catching the sunlight.",
    "A clear image of a blackboard with a clean, dark green surface and the word 'Hello' written precisely and legibly in the center with bold, white chalk letters.",
    "A minimalist photo of an orange tangerine with a green stem and leaves, symbolizing prosperity, sitting on a red silk cloth during Chinese New Year.",
    "A steaming cup of coffee on a wooden table.",
    "A glass of red wine on a reflective surface.",
    "A single drop of water clinging to a green leaf, with sunlight creating a faint rainbow pris",
    "An ancient stone bridge arching over a crystal-clear mountain stream, surrounded by lush greenery.",
    "A man riding a bicycle down a city street.",
    "A woman walking her dog in a park.",
    "A group of people playing soccer on a field.",
    "A child eating an ice cream cone at a carnival.",
    "A couple sitting on a bench near a lake.",
    "A chameleon blending in with a brown leaf.",
    "A chef preparing food in a busy kitchen.",
    "A boy flying a kite on a sunny day.",
    "A person reading a book in a library.",
    "A man surfing a large wave in the ocean.",
    "A woman painting on a canvas in her studio.",
    "A child playing with toys on the living room floor.",
    "A group of friends having a picnic in the park.",
    "A firefighter climbing a ladder during a rescue.",
    "A person holding an umbrella in the rain.",
    "A woman shopping for groceries at a market.",
    "A man fixing a bicycle in a workshop.",
    "A child riding a merry-go-round at an amusement park.",
    "A couple dancing together at a wedding.",
    "A person jogging along a forest trail.",
    "A teacher writing on a whiteboard in a classroom.",
    "A cute puppy",
    "A fluffy kitten",
    "A colorful parrot",
    "A majestic lion",
    "A playful dolphin",
    "A vibrant butterfly",
    "A glowing firefly",
    "A peaceful forest",
    "A bustling cityscape",
    "A graceful swan",
    "A bright sunflower",
    "a cute dog.",
    "a cute cat",
    "a bird flying on blue sky",
    "a house",
    "a horse",
    "A majestic snow leopard prowling through a snowy mountain terrain, its thick fur blending seamlessly with the icy landscape. Its piercing blue eyes stand out vividly, staring into the distance, with snowflakes gently falling around. The mountains in the background are draped in snow, creating a serene yet wild ambiance.",
    "An elegant Arabian horse galloping across a golden desert during sunset, its mane and tail flowing in the wind. The sand kicks up beneath its powerful hooves, and the sky is painted in shades of orange, pink, and purple. In the distance, the silhouette of sand dunes stretches into the horizon.",
    "A close-up of a regal Siberian tiger standing in a dense forest, its fur glowing under the dappled sunlight. The background shows a mix of green foliage and fallen autumn leaves, creating a perfect contrast with the tiger's orange and black stripes. Its eyes are focused, conveying both strength and grace.",
    "A portrait of an elderly woman in a traditional Moroccan outfit, sitting in front of a beautifully patterned tile wall. Her face is full of character, with deep wrinkles and a serene smile, and her hands rest on her lap. The background tiles display intricate geometric designs in vivid blues and greens, adding cultural richness to the image.",
    "A young ballet dancer practicing in an empty studio, her movements graceful and precise. She stands in a perfect arabesque pose, with her arms extended and her gaze focused in the mirror. The room is filled with soft natural light pouring through large windows, highlighting her dedication and discipline.",
    "A street musician playing a violin on a bustling city street, surrounded by a crowd of passersby. He is lost in his music, eyes closed, while people around him pause to listen. The cityscape behind him shows a mix of historic and modern buildings, with vibrant street lights beginning to illuminate as the evening sets in.",
    "A couple walking hand in hand through a vibrant autumn park. The trees are ablaze with shades of red, orange, and yellow, and leaves are gently falling around them. The couple wears cozy scarves and jackets, looking at each other with warmth and happiness. A path winds through the park, leading into the distance.",
    "A candid shot of a young girl laughing joyously, with wind blowing through her hair. She stands on a sandy beach, the ocean waves crashing in the background, and the sky is clear with just a hint of clouds. Her expression captures pure happiness and the carefree nature of childhood.",
    "A mythical dragon soaring above a misty mountain range, its scales glimmering in shades of emerald green and gold. Below, a dense forest stretches out, with ancient trees and hidden valleys. The dragon's wings are wide and powerful, casting a shadow over the land as it roars into the twilight sky.",
    "A medieval knight in shining armor standing atop a hill, looking down at a grand castle in the distance. The sky is dramatic with dark clouds rolling in, and the knight's cape flutters in the wind. The landscape around him is rugged and wild, with tall grasses and ancient trees.",
    "An enchanted castle built on the edge of a cliff, with waterfalls cascading down into a vast ocean below. The castle's spires reach high into the sky, surrounded by mist and clouds. Magical creatures like griffins and phoenixes fly around the castle, adding to the sense of wonder and mystery.",
    "A mermaid resting on a rocky shore, her tail shimmering in shades of turquoise and silver. The ocean waves crash gently against the rocks, and the sky is painted in soft hues of dawn. She gazes out toward the horizon with a look of longing, as seagulls fly above and the morning sun begins to rise.",
    "A bustling Tokyo street at night, illuminated by neon signs in various colors and shapes. The street is packed with people, some holding umbrellas as a light rain falls, creating reflections on the wet pavement. The atmosphere is electric, filled with the energy of the city and its vibrant nightlife.",
    "A serene canal scene in Venice at dawn, with gondolas gently floating on the water. The buildings along the canal are old and charming, with peeling paint and flower boxes on the windows. The sky is a soft pastel color as the sun rises, casting a golden glow over the entire scene.",
    "A futuristic city skyline at sunset, with skyscrapers of unique and imaginative designs. Some buildings have green terraces and rooftop gardens, while flying vehicles zoom between them. The sky is a blend of pink, orange, and deep purple, reflecting off the glassy surfaces of the buildings.",
    "A cozy café in Paris on a rainy day, with people sitting at small tables under a red awning. The street is wet, with reflections of the café's warm lights dancing on the pavement. A waiter in a black apron serves coffee to a couple, while the Eiffel Tower stands faintly visible in the background through the mist.",
    "A street market in Marrakech, bustling with colorful stalls selling spices, textiles, and ceramics. The air is filled with the aroma of exotic spices, and people in traditional attire barter over goods. The market is vibrant with rich colors, from the deep reds and yellows of the spices to the bright patterns of the fabrics.",
    "A serene alpine lake surrounded by towering snow-capped mountains, with crystal-clear water reflecting the peaks like a mirror. The shoreline is dotted with wildflowers in bloom, and a small wooden cabin sits on the edge, smoke rising from its chimney into the crisp morning air.",
    "A quiet village nestled in the valleys of Tuscany, surrounded by rolling hills and vineyards. Olive trees line the dirt roads leading to stone cottages with terracotta roofs. The sky is a clear blue with wispy clouds, and the entire scene feels warm and welcoming, bathed in the late afternoon light.",
    "An arid desert landscape at dusk, with dunes of golden sand stretching out into the horizon. A solitary cactus stands tall against the sky, which is ablaze with shades of orange, pink, and purple as the sun sets. The play of light and shadow on the dunes creates a dramatic, almost surreal effect.",
    "A close-up of a majestic African elephant walking through the savannah at sunset, its tusks gleaming in the warm light. In the background, acacia trees dot the landscape, and a herd of elephants follows in the distance. The sky is ablaze with colors, creating a powerful and serene moment.",
    "A vibrant parrot in the Amazon rainforest perched on a branch, its feathers displaying a stunning array of colors—reds, blues, greens, and yellows. The background is a blur of dense foliage, with glimpses of sunlight breaking through the leaves, highlighting the bird's intricate patterns.",
    "A lone wolf beneath shimmering northern lights.",
    "Sunlit meadow with wild poppies dancing in the breeze.",
    "A secret library lit by stained-glass lanterns.",
    "Rose-gold clouds floating over a lavender field.",
    "A moonlit castle high above rolling green hills.",
]      

captions.extend(captions_1)

def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""

    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image


def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
#    breakpoint()
   return [prompt]



def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



os.makedirs(os.path.join('/fsx/home/jiuhai.chen/interleaved-blip3u-2', f'visualization_{save_folder}'), exist_ok=True)



images = [
'/fsx/home/jiuhai.chen/interleaved-blip3u/getty_overfit/1001615122.jpg',
'/fsx/home/jiuhai.chen/interleaved-blip3u/getty_overfit/1006555508.jpg',
'/fsx/home/jiuhai.chen/interleaved-blip3u/getty_overfit/1019063528.jpg',
'/fsx/home/jiuhai.chen/interleaved-blip3u/getty_overfit/1024197308.jpg'
]

for img in images:
    set_global_seed(seed=42)
    gen_images = []
    inputs = add_template(["<image>\nPlease reconstruct the given image."])
    inputs.append(Image.open(f"{img}"))
    gen_img = pipe(inputs, guidance_scale=3.0)
    gen_images.append(gen_img.image)
    
    grid_image = create_image_grid(gen_images, 1, 1)
    basename = os.path.basename(img)
    number_str = os.path.splitext(basename)[0]
    grid_image.save(os.path.join('/fsx/home/jiuhai.chen/interleaved-blip3u-2', f'visualization_{save_folder}', f"{number_str}_0.png"))








for prompt in captions:
    set_global_seed(seed=42)
    gen_images = []
    for i in range(4):
        gen_img = pipe(add_template([f"Please generate image based on the following caption: {prompt}"]), guidance_scale=3.0)
        gen_images.append(gen_img.image)
    print(f"finish {prompt}")

    
    grid_image = create_image_grid(gen_images, 2, 2)
    grid_image.save(os.path.join('/fsx/home/jiuhai.chen/interleaved-blip3u-2', f'visualization_{save_folder}', f"{prompt[:100]}_{4}.png"))



