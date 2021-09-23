# yolov5s.pt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gnlucZtQji5n5ZaxPcl2Nd_w11fHZ8pu' -O yolov5s.pt

# model.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_LoiFYlsVu3ervIidYIMopodyBLswmi4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_LoiFYlsVu3ervIidYIMopodyBLswmi4" -O model.pth && rm -rf /tmp/cookies.txt

# resnet101_8x8f_denseserial.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11_-anDRcX9f7bCd9Xg_eh0Gp2fajCGZj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11_-anDRcX9f7bCd9Xg_eh0Gp2fajCGZj" -O resnet101_8x8f_denseserial.pth && rm -rf /tmp/cookies.txt

# yolov3-spp.weight
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-D-bk-8XuUK4n3Vfv8YU0EdlZsH2y0I8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-D-bk-8XuUK4n3Vfv8YU0EdlZsH2y0I8" -O yolov3-spp.weights && rm -rf /tmp/cookies.txt