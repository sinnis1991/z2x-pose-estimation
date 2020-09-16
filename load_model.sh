#!/bin/sh

if [ $1='1' ]
then
  #https://drive.google.com/file/d/1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs/view?usp=sharing
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
    /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs' \
    -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs" -O model.tar && rm -rf /tmp/cookies.txt

  tar -xvf model.tar
  mv ./model_1 $2
elif [ $1='3' ]
then
	#https://drive.google.com/file/d/1-UdE8ZUe4BVELN86OWLt5jAcCGv-Stsz/view?usp=sharing
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
    /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-UdE8ZUe4BVELN86OWLt5jAcCGv-Stsz' \
    -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-UdE8ZUe4BVELN86OWLt5jAcCGv-Stsz" -O model.tar && rm -rf /tmp/cookies.txt

  tar -xvf model.tar
  mv ./model_3 $2
fi