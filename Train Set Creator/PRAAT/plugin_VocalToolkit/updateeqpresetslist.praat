s = selected("Sound")
Create Strings as file list... list eq/*.Sound
numberOfFiles = Get number of strings
i=1
for i to numberOfFiles
	fileName_'i'$ = Get string... i
endfor
Remove
select s
i=1
txt$ = ""
for i to numberOfFiles
	fileNameTmp$ = fileName_'i'$
	l = length(fileNameTmp$)-6
	fileName$ = left$(fileNameTmp$,l)
	txt$ = "'txt$'option 'fileName$''newline$'"
endfor
txt$ > eqpresetslist.txt
pause EQ Presets list has been updated.
