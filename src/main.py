from medilens_utils.general import BoundingBox, Image, Point, Ellipse
from medilens_utils import APIClient
import os
import shutil
if os.path.exists("results"):
    shutil.rmtree("results")
    
img = Image("data/cad/some_pixels/01.png")
img = Image("data/0_jpg.rf.3b701d12729b82b2c90b71e563b170d1.jpg")

api_client = APIClient("https://medilens.deife.tech")
detections = api_client.get_pill(img)

for i, (nw, se) in enumerate(detections):
    ne = Point(se.x, nw.y)
    sw = Point(nw.x, se.y)
    bb = BoundingBox(
        nw,
        ne,
        se,
        sw
    )
    print(ne, sw)
    img.cut(nw, se)
    img.save_at(f"results", f"{i}.png")
    img.reload()

img = Image("results/0.png")
img = Image("data/cad/corners_only/13.png")
ellipse = api_client.get_ellipse(img)
print(ellipse)
img.draw_cross(Point(int(ellipse.cx), int(ellipse.cy)))
img.draw_ellipse(ellipse)
img.save_at("results", "ellipse.png")
img.reload()