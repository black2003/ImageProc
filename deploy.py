import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
def showimg(awe,strb):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(awe,strb)
    plt.axis('off')
    st.pyplot(fig)



st.warning("Please Uplode the image here")
upload= st.file_uploader('Insert image for Image Processing', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
    im= Image.open(upload)
    img= np.asarray(im)
    img = cv2.resize(img,(224, 224))
    c1.header('Input Image')
    c1.image(im)
    c1.write(img.shape)
    grt = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    st.success("Greyscaled image")
    showimg(grt,"gray")
    st.success("Blured image")
    blur = cv2.GaussianBlur(grt,(5,5),0)
    showimg(blur,"gray")
    st.success("Sobel Applied image")
    sobel_meth = st.selectbox(
        "Select Method:",
        ("Edge x", "Edge y","Combined")
    )
    sbksz = st.slider("k-size",1,5,2)
    if sobel_meth == "Edge x":
        edges1 = cv2.Sobel(grt,cv2.CV_32F,1,0,ksize=sbksz)
        showimg(edges1,"gray")
    elif sobel_meth == "Edge y":
        edges2 = cv2.Sobel(grt,cv2.CV_32F,0,1,ksize=sbksz)
        showimg(edges2,"gray")
    elif sobel_meth == "Combined":
        edges1 = cv2.Sobel(grt,cv2.CV_32F,1,0,ksize=sbksz)
        edges2 = cv2.Sobel(grt,cv2.CV_32F,0,1,ksize=sbksz)
        we = cv2.add(edges1,edges2)
        showimg(we,"gray")
    st.success("Canny Applied Image")
    st.warning("Hyper Parameter Tuning for Canny Filter")
    xdif = st.slider("X-diff", 1,500,1)
    ydif = st.slider("Y-diff",1,500,1)
    if st.button("Show Canny"):
        edge3 = cv2.Canny(blur,xdif,ydif)
        showimg(edge3,"gray")
    st.warning("Apply Median Blur for Thresh Img")
    ksz = st.slider("K-Size",1,5,2)
    img_f = cv2.medianBlur(img, ksz)
    grf = cv2.cvtColor(img_f,cv2.COLOR_BGR2GRAY)
    if st.button("Show Median"):
        showimg(grf,"gray")
    st.success("Thresh Applied Image")
    thresh_meth = st.selectbox(
        "Select Method:",
        ("Only Thresh Binary", "Only Thresh Binary Inverse","Thresh Binary Otsu","Thresh Binary_Inv Otsu")
    )
    if thresh_meth == "Only Thresh Binary":
        (T, threshInv) = cv2.threshold(grf,0, 255,cv2.THRESH_BINARY)
        showimg(threshInv,"gray")
    elif thresh_meth == "Only Thresh Binary Inverse":
        (T, threshInv) = cv2.threshold(grf,0, 255,cv2.THRESH_BINARY_INV)
        showimg(threshInv,"gray")
    elif thresh_meth == "Thresh Binary Otsu":
        (T, threshInv) = cv2.threshold(grf,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        showimg(threshInv,"gray")        
    elif thresh_meth == "Thresh Binary_Inv Otsu":
        (T, threshInv) = cv2.threshold(grf,0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        showimg(threshInv,"gray")
    
    st.warning("Kernal Shape for further process")
    xaxs = st.slider("kernal row",1,6,1)
    yaxs = st.slider("kernal column",1,6,1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(xaxs,yaxs))
    st.warning("Define the iterations")
    ite = st.slider("No of Iterations",1,7,1)
    
    Morph_meth = st.selectbox(
        "Select Method:",
        ("Erosion", "Dilation","Opening","Closing")
    )
    if Morph_meth == "Erosion":
        bkr = cv2.erode(threshInv,k,iterations = ite)
    elif Morph_meth == "Dilation":
        bkr = cv2.dilate(threshInv,k,iterations = ite)
    elif Morph_meth == "Opening":
        bkr = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, k,iterations=ite)
    elif Morph_meth == "Closing":
        bkr = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, k,iterations=ite)

    showimg(bkr,"gray")
    
    B1 = cv2.morphologyEx(threshInv,cv2.MORPH_OPEN,k,iterations=1)
    
    sure_bg = cv2.dilate(B1,k,iterations=1)
    if st.button("Show Background"):
        showimg(sure_bg,"gray")
    
    D = cv2.distanceTransform(B1,cv2.DIST_L2,0)
    if st.button("Show Distance Transform"):
        showimg(D,"gray")
    
    disr = st.slider("Distance fg set",0.01,0.1,0.01)
    
    r2, sure_fg_img = cv2.threshold(D, disr*D.max(),255,cv2.THRESH_BINARY)
    
    sure_fg = sure_fg_img.astype(np.uint8)
    if st.button("Show Foreground"):
        showimg(sure_fg,"gray")
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    if st.button("Show Unknown"):
        showimg(unknown,"gray")
    
    r3,M = cv2.connectedComponents(sure_fg)
    M = M +1  
    M[unknown==255]=0
    if st.button("Show Connected Components"):
        showimg(M,"tab20")
    
    imgnext = img

    M2 = cv2.watershed(imgnext,M)
    st.success("Success Watershed")
    showimg(M2,"tab20")

    

    L = np.unique(M2)
    New_img=[]
    for l in L[2:]:
        T = np.where(M2==l,255,0).astype(np.uint8)
        c, h = cv2.findContours(T,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        New_img.append(c[0])
    
    thk = st.slider("Define Contour Thickness",1,10,1)
    imgnext = cv2.drawContours(imgnext,New_img,-1,color=(255,0,0),thickness=thk)
    st.success("Showing the Segmented image")
    fig = plt.figure()
    plt.imshow(imgnext)
    plt.axis('off')
    plt.title("Segmentation")
    st.pyplot(fig)
    
    
else:
    st.error("Image is not Uploaded, Try re uploading")

