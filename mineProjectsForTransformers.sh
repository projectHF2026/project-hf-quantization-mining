#!/bin/bash

#filename='dependent_trasformers.csv'
filename='tryal.csv'
n=1
resultsDir='grepResults/'

mkdir $resultsDir

while read line; do
# reading each line
echo "Line No. $n : $line"

# retrieving prj name (/owner/prj_name/)
prj=(`echo $line | cut -d "," -f 1`)
#echo "$prj"

# selecting dirName and owner of a prj
dirName=(`echo $prj | cut -d "/" -f 3`)
owner=(`echo $prj | cut -d "/" -f 2`)
#echo "$dirName"

# creating github.com URL to clone the project
gHttps='https://'
gitHubURL='@github.com'
username='USERNAME'
password='PASSWORD'
credentials=(`echo $username:$password`)
prjPath=(`echo $gHttps$credentials$gitHubURL$prj`)
echo $prjPath

# cloning the project
git clone --depth=1 $prjPath

# creating the name of file for grep results 
ext='.txt'
fileGrep=(`echo "$owner""£sep£""$dirName"`)

# grep the string into project directory and storing results into a file
grep -RiI ".from_pretrained" ./$dirName > "$resultsDir$fileGrep$ext"

# remove dir of cloned project
dotSlash='./'
rm -rf $dotSlash$dirName

remainder=$(( n % 100 ))
if [ "$remainder" -eq 0 ]; then
	sleep 1800
fi

#echo "grep -RiI ".from_pretrained" ./$dirName/"
n=$((n+1))
done < $filename