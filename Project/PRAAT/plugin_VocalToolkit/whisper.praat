include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	tmp = selected("Sound")
	Scale... 0.9999
	execute gate.praat 20
	tmp2 = selected("Sound")
	sr = Get sample rate
	dur = Get total duration
	bgnoise = Create Sound from formula... bgnoise 1 0 'dur' 'sr' randomUniform(-1, 1)
	Scale intensity... 0.01
	select tmp2
	Formula...  self + Object_'bgnoise'[]
	pred_order = round(sr/1000) + 2
	lpc = To LPC (burg)... pred_order 0.025 0.01 50
	noise = Create Sound from formula... noise 1 0 'dur' 'sr' randomUniform(-1, 1)
	plus lpc
	tmp3 = Filter... yes
	execute workpost.praat
	tmp4 = selected("Sound")
	execute eq10bands.praat no 1 -24 -24 -24 -24 12 24 24 12 12 -6
	result = Rename... 's$'_whisper
	execute fixdc.praat
	select tmp
	plus tmp2
	plus bgnoise
	plus lpc
	plus noise
	plus tmp3
	plus tmp4
	Remove
	select result
endproc
