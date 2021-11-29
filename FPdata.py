from google_images_download import google_images_download
import os
import shutil

# Crawling
def imageCrawling(keyword, dir):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keyword,
                 "limit": 100,
                 "print_urls": False,
                 "no_directory": True,
                 "output_directory": dir}
    paths = response.download(arguments)
    print(paths)


# imageCrawling('紫禁城', './data')
# imageCrawling('中国古城 ', './data')
# imageCrawling('상하이 예원', './data')
# imageCrawling('자금성', './data')
# imageCrawling('중국 이화원', './data')
# imageCrawling('천안문', './data')
# imageCrawling('중국 천후궁', './data')
# imageCrawling('중국 고성', './data')


# imageCrawling('', './data')

# New naming for each dataset
base = './data'
if not os.path.isdir(base):
    os.makedirs(base)



