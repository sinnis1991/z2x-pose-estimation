#!/bin/sh

if [ $1='1' ]
then
  #https://drive.google.com/file/d/1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs/view?usp=sharing
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
    /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs' \
    -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wJUWcsITnY3YAyGdUuS54PxH7E3AJIEs" -O model.tar && rm -rf /tmp/cookies.txt

  tar -xvf model.tar
  mv ./model_1 $2
fi

# echo $2