"""Crop a vector plot PDF to its rendered ink bounds, leaving a 4pt margin."""
from pathlib import Path
import sys, subprocess
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
p=Path(sys.argv[1])
im=Image.open(p.with_suffix('.png')).convert('RGB')
mask=ImageChops.difference(im,Image.new('RGB',im.size,'white')).convert('L').point(lambda x:255 if x>30 else 0)
x0,y0,x1,y1=mask.getbbox()
page=PdfReader(p).pages[0]; w=float(page.mediabox.width);h=float(page.mediabox.height)
page.mediabox.lower_left=(max(0,x0/im.width*w-4),max(0,(1-y1/im.height)*h-4))
page.mediabox.upper_right=(min(w,x1/im.width*w+4),min(h,(1-y0/im.height)*h+4))
page.cropbox=page.mediabox
writer=PdfWriter();writer.add_page(page)
tmp=p.with_suffix('.cropped.pdf')
with tmp.open('wb') as f:writer.write(f)
tmp.replace(p)
subprocess.run(['pdftoppm','-png','-r','200','-singlefile',str(p),str(p.with_suffix(''))],check=True)
