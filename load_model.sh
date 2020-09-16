#!/bin/sh

a="model1"
c="model3"

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
# echo $c
else
echo "invalid index"
fi