from PIL import Image, ImageDraw, ImageFont

def generate_tiles(n=4, size=210, font_size=120, output_dir="tiles"):
    """
    Generate numbered square PNG tiles with alternating red and gray backgrounds.
    
    Args:
        n (int): number of tiles to generate (1..n).
        size (int): width/height of each tile in pixels.
        font_size (int): size of the number font.
        output_dir (str): folder to save images.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # colors
    red = (220, 100, 100)   # soft red
    gray = (160, 160, 160)  # soft gray
    colors = [red, gray]    # alternating

    # load font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for i in range(1, n+1):
        bg_color = colors[i % 2 == 0]  # odd=red, even=gray
        img = Image.new("RGB", (size, size), bg_color)
        draw = ImageDraw.Draw(img)
        text = str(i)

        # measure text bbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # calculate perfect center (taking ascent into account)
        x = (size - text_w) // 2 - bbox[0]
        y = (size - text_h) // 2 - bbox[1]

        draw.text((x, y), text, fill="white", font=font)

        img.save(f"{output_dir}/slidetile_{i}.png")

    print(f"Generated {n} tiles in '{output_dir}'.")

# Example
if __name__ == "__main__":
    generate_tiles(n=16)

