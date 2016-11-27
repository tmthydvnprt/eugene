#!/usr/bin/bash

# test project
nosetests tests/test_eugene.py -v -d --with-coverage --cover-package=eugene,tests --cover-tests --cover-erase --cover-inclusive --cover-html --cover-branches &> eugene.txt.temp

# test report
echo '' > test_report.txt
echo 'Eugene Testing Report' >> test_report.txt
echo `date` >> test_report.txt
echo '=========================================' >> test_report.txt

echo '' >> test_report.txt
cat eugene.txt.temp >> test_report.txt

rm .coverage
rm eugene.txt.temp

echo 'project tested'
