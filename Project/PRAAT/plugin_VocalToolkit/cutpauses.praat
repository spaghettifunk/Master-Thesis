include batch.praat
procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	tg = nowarn nocheck To TextGrid (silences)...  20 0 -25 0.1 0.1 pause *
	plus wrk
	nowarn nocheck Extract intervals where... 1 no "is not equal to" pause
	ns = numberOfSelected("Sound")
	ntg = numberOfSelected("TextGrid")
	if ns = 1 and ntg = 0
		if selected("Sound") = wrk
			Remove
			select s
			Copy... 's$'_cutpauses
		else
			tmp = selected("Sound")
			execute workpost.praat
			result = Rename... 's$'_cutpauses
			select wrk
			plus tg
			plus tmp
			Remove
			select result
		endif
	endif
	if ns = 1 and ntg = 1
		Remove
		select s
		Copy... 's$'_cutpauses
	endif
	if ns > 1 and ntg = 0
		for i to ns
			pau'i' = selected("Sound", i)
		endfor
		tmp = Concatenate
		execute workpost.praat
		result = Rename... 's$'_cutpauses
		select wrk
		plus tg
		for i to ns
			plus pau'i'
		endfor
		plus tmp
		Remove
		select result
	endif
endproc
