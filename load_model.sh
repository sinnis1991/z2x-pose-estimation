#!/bin/sh

a="model1"
c="model3"
d="model4"
e="model5"

if [ "$1" == "$a" ]
then
#https://drive.google.com/file/d/1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_1 $2
# echo $a
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
# echo $c
# echo $c
elif [ "$1" == "$e" ]
then
# https://drive.google.com/file/d/1h06ozWwQAsyW4IZeZgKXO5tTAUQ1U4Pf/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h06ozWwQAsyW4IZeZgKXO5tTAUQ1U4Pf' \
  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h06ozWwQAsyW4IZeZgKXO5tTAUQ1U4Pf" -O model.tar && rm -rf /tmp/cookies.txt

tar -xvf model.tar
mv ./model_5 $2
# echo $c
else
echo "invalid index"
fi