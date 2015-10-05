form Change note
	optionmenu Note 1
		option C       Do
		option C#     Do#
		option D       Re
		option D#     Re#
		option E       Mi
		option F       Fa
		option F#     Fa#
		option G       Sol
		option G#     Sol#
		option A       La
		option A#     Si b
		option B       Si
	choice Octave 1
		button -1
		button 1
		button +1
	boolean Monotone 0
endform
include batch.praat
procedure action
	if note = 1
		note$ = "do"
		if octave = 1
			note = 130.81
		elsif octave = 2
			note = 261.63
		elsif octave = 3
			note = 523.25
		endif
	elsif note = 2
		note$ = "do_s"
		if octave = 1
			note = 138.59
		elsif octave = 2
			note = 277.18
		elsif octave = 3
			note = 554.36
		endif
	elsif note = 3
		note$ = "re"
		if octave = 1
			note = 146.83
		elsif octave = 2
			note = 293.66
		elsif octave = 3
			note = 587.33
		endif
	elsif note = 4
		note$ = "re_s"
		if octave = 1
			note = 155.56
		elsif octave = 2
			note = 311.13
		elsif octave = 3
			note = 622.25
		endif
	elsif note = 5
		note$ = "mi"
		if octave = 1
			note = 164.81
		elsif octave = 2
			note = 329.63
		elsif octave = 3
			note = 659.25
		endif
	elsif note = 6
		note$ = "fa"
		if octave = 1
			note = 174.61
		elsif octave = 2
			note = 349.23
		elsif octave = 3
			note = 698.46
		endif
	elsif note = 7
		note$ = "fa_s"
		if octave = 1
			note = 185.99
		elsif octave = 2
			note = 369.99
		elsif octave = 3
			note = 739.99
		endif
	elsif note = 8
		note$ = "sol"
		if octave = 1
			note = 196
		elsif octave = 2
			note = 391.99
		elsif octave = 3
			note = 783.99
		endif
	elsif note = 9
		note$ = "sol_s"
		if octave = 1
			note = 207.65
		elsif octave = 2
			note = 415.30
		elsif octave = 3
			note = 830.61
		endif
	elsif note = 10
		note$ = "la"
		if octave = 1
			note = 220
		elsif octave = 2
			note = 440
		elsif octave = 3
			note = 880
		endif
	elsif note = 11
		note$ = "si_b"
		if octave = 1
			note = 233.08
		elsif octave = 2
			note = 466.16
		elsif octave = 3
			note = 932.33
		endif
	elsif note = 12
		note$ = "si"
		if octave = 1
			note = 246.94
		elsif octave = 2
			note = 493.88
		elsif octave = 3
			note = 987.77
		endif
	endif
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
include minmaxf0.praat
	if monotone
		tmp = Change gender... minF0 maxF0 1 'note' 0 1
	else
		tmp = Change gender... minF0 maxF0 1 'note' 1 1
	endif
	execute workpost.praat
	if monotone
		result = Rename... 's$'_changenote_'note$'_monotone
	else
		result = Rename... 's$'_changenote_'note$'
	endif
	select wrk
	plus tmp
	Remove
	select result
endproc
