#!/bin/sh

a="model1"
b="model2"
c="model3"
d="model4"
e="model5"
f="model6"

if [ "$1" == "$a" ]
then
#https://drive.google.com/file/d/1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_1 $2
elif [ "$1" == "$b" ]
then
# https://drive.google.com/file/d/1W41dDSDl3dwMpOc7BPqmVKlSdHM6cuss/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W41dDSDl3dwMpOc7BPqmVKlSdHM6cuss' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W41dDSDl3dwMpOc7BPqmVKlSdHM6cuss" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_2 $2
elif [ "$1" == "$c" ]
then
# https://drive.google.com/file/d/1-UdE8ZUe4BVELN86OWLt5jAcCGv-Stsz/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-UdE8ZUe4BVELN86OWLt5jAcCGv-Stsz' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-UdE8ZUe4BVELN86OWLt5jAcCGv-Stsz" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_3 $2
elif [ "$1" == "$d" ]
then
# https://drive.google.com/file/d/1ntn3QIIJjgCr0N5ECv2DoLdYWRPLUW3P/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ntn3QIIJjgCr0N5ECv2DoLdYWRPLUW3P' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ntn3QIIJjgCr0N5ECv2DoLdYWRPLUW3P" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_4 $2
elif [ "$1" == "$e" ]
then
# https://drive.google.com/file/d/1h06ozWwQAsyW4IZeZgKXO5tTAUQ1U4Pf/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h06ozWwQAsyW4IZeZgKXO5tTAUQ1U4Pf' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h06ozWwQAsyW4IZeZgKXO5tTAUQ1U4Pf" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_5 $2
elif [ "$1" == "$f" ]
then
# https://drive.google.com/file/d/1exMdK7_qJdVlq_fh-FN_dB_XnO9pZQWM/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1exMdK7_qJdVlq_fh-FN_dB_XnO9pZQWM' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1exMdK7_qJdVlq_fh-FN_dB_XnO9pZQWM" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_6 $2
else
echo "invalid index"
fi