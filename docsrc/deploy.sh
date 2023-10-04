make clean
make html
rm -rf ../docs
cp -r build/html ../
mv ../html ../docs
