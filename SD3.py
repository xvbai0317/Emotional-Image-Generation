import torch
from diffusers import StableDiffusion3Pipeline

def generate_image(prompt, image_path):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]

    image.save(image_path)
# 使用示例
# prompt = """### Picture composition:
# - ** Foreground ** : You can design a vast field or mountain path, but unlike the common peaceful countryside, the scene here is slightly chaotic. The trees seemed to have been battered by the wind, their branches twisted, and they showed a state of struggle and defiance.
# - ** Medium scene ** : The introduction of a river or stream, the current is swift, symbolizing the inner torrent and the release of emotions. The color of the river tends to be dark blue or dark green, with a gloomy feeling, but without losing vitality.
# - ** Long view ** : The mountains are undulating and shrouded in clouds, but the peaks appear more sharp and steep, as if they are venting anger. Clouds in the sky are thick and moving quickly, adding to the dynamic feeling of the picture.
#
# ### Color application:
# - The main color uses a strong contrast between warm and cold colors, such as dark blue, dark green, black and other cool colors, representing anger and depression; At the same time, a small amount of orange-red or yellow as an embellishment, symbolizing the energy and passion when anger bursts.
# - The water body can use a blue gradient, from light blue to dark blue to black, to show the change of water potential and inner fluctuations.
# - The land and vegetation are used in heavy green or brown to add weight to the picture.
#
# ### Element details:
# - Add lightning, storms and other natural phenomena in some specific locations to enhance the emotional atmosphere.
# - The branches of trees can be designed to twist and show a gesture of struggle and resistance.
# - One or two hidden figures can be added in a corner, they may stand or kneel, facial expression is not obvious, but the body posture reveals the angry message, so that the viewer can sympathize with the setting."""
# image_path = './imgs/Qwen.png'