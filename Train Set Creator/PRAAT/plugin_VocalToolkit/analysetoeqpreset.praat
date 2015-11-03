form Analyse to EQ Preset
	word New_EQ_Preset_Name
endform
if new_EQ_Preset_Name$ = ""
	exit Please enter some EQ Preset name
endif
s = selected("Sound")
execute ireq.praat
Save as binary file... eq/'new_EQ_Preset_Name$'.Sound
Remove
call UpdateEQlist
select s
pause New EQ Preset has been saved as "'new_EQ_Preset_Name$'".
procedure UpdateEQlist
	Create Strings as file list... list eq/*.Sound
	numberOfFiles = Get number of strings
	i=1
	for i to numberOfFiles
		fileName_'i'$ = Get string... i
	endfor
	Remove
	i=1
	txt$ = ""
	for i to numberOfFiles
		fileNameTmp$ = fileName_'i'$
		l = length(fileNameTmp$)-6
		fileName$ = left$(fileNameTmp$,l)
		txt$ = "'txt$'option 'fileName$''newline$'"
	endfor
	txt$ > eqpresetslist.txt
endproc
