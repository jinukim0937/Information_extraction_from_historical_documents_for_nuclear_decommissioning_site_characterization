def paper_ocr(paper_name):
    from pdf2image import convert_from_path
    from PIL import Image
    import numpy as np
    import easyocr
    
    #reader = easyocr.Reader(['en','ko'], gpu = True,
    #               model_storage_directory='./user_network_dir',
    #                user_network_directory='./user_network_dir',
    #                recog_network='custom')# 'en': 영어로 설정
    
    reader = easyocr.Reader(['en','ko'], gpu = True)# 'en': 영어로 설정
    
    # .txt 파일이라면
    if paper_name[-3:] == 'txt':
        with open(paper_name, 'r', encoding='utf-8') as f:
            paper_temp = f.read()
            paper_temp = [[paper_temp]]
            
    # .pdf 파일이라면
    elif paper_name[-3:] == 'pdf':
        with open(paper_name, "rb") as f:    
            # PDF 파일을 이미지로 변환
            pages = convert_from_path(paper_name, 500, poppler_path=r".\poppler-23.01.0/Library/bin")
            # 이미지에서 텍스트 추출
            paper_temp = []
            for page in pages:
                image_array = np.array(page)
                result = reader.readtext(image_array, detail=0, paragraph=1)
                paper_temp.extend([result])
            
    return paper_temp