# OCR

Optical Character Recognition (OCR) is a current practice in Computer Vision. Several applications use OCR, for example, extraction of document information in the medical field, novices, and technical documentation. We can also use OCR for smart city applications, like traffic control of matriculation plates. Another point to know about OCR is the fact that we are also using the Natural Language Process (NLP). OCR is Computer Vision and NLP.
 
The blog of [PyImageSearch](https://pyimagesearch.com/?s=ocr) introduces many techniques widely used to perform OCR. This repository is the result of a grabbing into the site. The codes of all examples posted are templates from PyImageSearch, but with proper modifications.

In total, this repository contains 10 folders. Each of them is about a specific subject of OCR. The main liraries used here are PyTesseract (built on Tesseract) and OpenCV. The repository is organized as follow:

1 - [Simple OCR](https://github.com/IgorMeloS/OCR/tree/main/1%20-%20Simple%20OCR)
    
   The folder contains the file [simple_OCR.ipynb](https://github.com/IgorMeloS/OCR/blob/main/1%20-%20Simple%20OCR/simple_OCR.ipynb), which is an introduction to OCR using PyTessaract (the main options to OCR simple documents).

2 - [Digts Recognition](https://github.com/IgorMeloS/OCR)

  The file [digts.ipynb](https://github.com/IgorMeloS/OCR/blob/main/2%20-%20Digts%20Recognition/digts.ipynb) shows how to recognize digits using PyTesseract and introduces the concept of white and black list.
  
3 - [Text Orientation](https://github.com/IgorMeloS/OCR/tree/main/3%20-%20Text%20Orientation)
 
   Text orientation can be considered a pre-processing stage when building an OCR engine. In the file [text_orientation.ipynb](https://github.com/IgorMeloS/OCR/blob/main/3%20-%20Text%20Orientation/text_orientation.ipynb), we find how to perform text orientation using PyTesseract.

4 - [Translation with Tesseract](https://github.com/IgorMeloS/OCR/tree/main/4%20-%20Translation%20with%20Tesseract)

  Document translation can be performed using OCR for several languages. In the file [OCR_translater.ipynb](https://github.com/IgorMeloS/OCR/blob/main/4%20-%20Translation%20with%20Tesseract/OCR_translater.ipynb), we find an application to translate text. To perform it, we need to consider two essential libraries. The first is PyTesseract which will localize the text in the image (note: we need to set our OCR engine to recognizer non-English languages). The second library used is the TextBlob that performs the translations. In this example, we consider three languages: English, French, and Portuguese. You'll find two poems used for the test.
 
5 - [Image pre-processing](https://github.com/IgorMeloS/OCR/tree/main/5%20-%20Image%20pre-processing)

  Until now, we just have seen examples with ideal images. When we are confronted with real-word problems, images might present imperfections like noise, for example. PyTesseract works well given "good" images, on the other hand, the results can be not satisfactory. Using OpenCV, we can pre-process images eliminating the excess of information. In the file [image-preprocessing.ipynb](https://github.com/IgorMeloS/OCR/blob/main/5%20-%20Image%20pre-processing/image-preprocessing.ipynb), we find the proper image pre-process based on the [StackOverFlow question](https://stackoverflow.com/questions/33881175/remove-background-noise-from-image-to-make-text-more-clear-for-ocr).
  
6 - [MZR](https://github.com/IgorMeloS/OCR/tree/main/6%20-%20MZR)
  
  Machine Zone Readable for the passports is another application of OCR. In this case, we consider some "toy passports" to extract the desired information. Once again, PyTesseract is not able to recognizer characters due to the image is not enough cleaned. We must pre-process the image with OpenCV to extract the region of interest and then use Pytesseract to recognize. You find the model in the file [mzr.ipynb](https://github.com/IgorMeloS/OCR/blob/main/6%20-%20MZR/mzr.ipynb)

7 - [Template-matching](https://github.com/IgorMeloS/OCR/tree/main/7%20-%20template-matching)

  Templating matching is a technique to recognize characters in text from a pre-defined template. In this example, we explore the function template matching from OpenCV to recognize digits from Credit Card. The image template chosen for this task is the OCR-A font template. The results given by the model are the numbers and the kind of card, for example, Visa or MasterCard. Technical like this can offer efficient and practical scanners. You find the full model in the file [template-matching.ipynb](https://github.com/IgorMeloS/OCR/blob/main/7%20-%20template-matching/template-matching.ipynb).
 
8 - [Text Bounding Box](https://github.com/IgorMeloS/OCR/tree/main/8%20-%20Text%20Bounding%20Box)

  Using PyTesseract, we can recognize text in images. But if we want to draw the bounding box around the world is it possible? The answer is yes. With PyTessaract, we can obtain the bounding boxes to each detection and its confidence. For this example, we consider a simple image proposed by PyImageSearch. Additionally, I present an example of a noised image that requires a strong image pre-processing. You find the entire code in the file [text_bounding_box.ipynb](https://github.com/IgorMeloS/OCR/blob/main/8%20-%20Text%20Bounding%20Box/text_bounding_box.ipynb). I posted the same method on [StackOverflow](https://stackoverflow.com/questions/70007353/difficulty-detecting-digits-with-tesseract/70020559#70020559).
  
9 - [Text Bounding Box with OpenCV (EAST)](https://github.com/IgorMeloS/OCR/tree/main/9%20-%20Text%20Bounding%20Box%20with%20OpenCV%20(EAST))

  Detecting text bounding boxes with PyTesseract imposes some limitations. For example, in a natural scene, some texts are rotated or vertical, these texts are not detected. To tackle this problem, we can use a pre-trained model with OpenCV. A related model to this task is [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155). The model requires more lines of code than other examples exposed here. Generally speaking, we must detect and post-process the detections. There's a little module to post-process all detections, [east](https://github.com/IgorMeloS/OCR/tree/main/9%20-%20Text%20Bounding%20Box%20with%20OpenCV%20(EAST)/east). The [East_and_Tessseract.ipynb](https://github.com/IgorMeloS/OCR/blob/main/9%20-%20Text%20Bounding%20Box%20with%20OpenCV%20(EAST)/East_and_Tessseract.ipynb) contains the entire model.
  
10 - [Text Detection and recognition](https://github.com/IgorMeloS/OCR/tree/main/10%20-%20Text%20Detection%20and%20recognition)

  The combination of the EAST model and Tesseract is seminal for OCR tasks. While Tesseract can fail to detect rotated texts, the EAST model does it well. On the other hand, the EAST model only detects texts. To recognize the texts we must use Tesseract. This example is a complete guide on how to merge the EAST model and Tessaract. This technology can be used for a considerable number of OCR problems. Each problem has its specificities and demands changes in the code. The code is finde in the file [Detect_and_OCR.ipynb](https://github.com/IgorMeloS/OCR/blob/main/10%20-%20Text%20Detection%20and%20recognition/Detect_and_OCR.ipynb).
  
## Conclusions

The examples exposed here can be considered as a start point to deploy OCR models. Even if the cases are simple, most of the techniques used here can be deployed in a real-world problem. But the great conclusion is, we must have a model to detect and another model to recognize texts. Which model do we need to use? The task will require.
