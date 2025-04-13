import fitz
import re
import os

file_path = r'graduate-handbook.pdf' # PDF 文件路径
dir_path = r'output_images' # 存放图片的文件夹

def pdf2image1(path, pic_path):
    checkIM = r"/Subtype(?= */Image)"
    pdf = fitz.open(path)
    lenXREF = pdf.xref_length()
    count = 1
    for i in range(1, lenXREF):
        text = pdf.xref_object(i)
        isImage = re.search(checkIM, text)
        if not isImage:
            continue
        pix = fitz.Pixmap(pdf, i)
        if pix.size < 10000:  # 在这里添加一处判断一个循环
            continue  # 不符合阈值则跳过至下
        new_name = f"img_{count}.png"
        pix.save(os.path.join(pic_path, new_name))
        # pix.writePNG()
        count += 1
        pix = None

pdf2image1(file_path, dir_path)