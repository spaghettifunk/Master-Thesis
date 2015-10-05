include batch.praat
procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	sr = Get sample rate
include minmaxf0.praat
	pitch = To Pitch... 0.01 minF0 maxF0
	threshold = 21
	select wrk
	if sr>11025
		downsampled = Resample... 11025 1
	else
		downsampled = Copy... tmp
	endif
	Filter with one formant (in-line)...  1000 500
	framelength = 0.01
	int_tmp = To Intensity... 40 'framelength' 0
	maxint = Get maximum... 0 0 Cubic
	t1 = Get time from frame... 1
	matrix_tmp = Down to Matrix
	endtime = Get highest x
	ncol = Get number of columns
	coldist = Get column distance
	h=1
	newt1 = 't1'+('h'*'framelength')
	ncol = 'ncol'-(2*'h')
	matrix_intdot = Create Matrix... intdot 0 'endtime' 'ncol' 'coldist' 'newt1' 1 1 1 1 1 (Object_'matrix_tmp'[1,col+'h'+'h']-Object_'matrix_tmp'[1,col]) / (2*'h'*dx)
	temp_IntDot = To Sound (slice)... 1
	temp_rises = To PointProcess (extrema)... Left yes no Sinc70
	select temp_IntDot
	temp_peaks = To PointProcess (zeroes)... Left no yes
	npeaks = Get number of points
	select downsampled
	plus matrix_tmp
	plus matrix_intdot
	plus temp_IntDot
	Remove
	cnt = 1
	for pindex from 1 to 'npeaks'
		select temp_peaks
		ptime = Get time from index... 'pindex'
		select int_tmp
		pint = Get value at time... 'ptime' Nearest
		select pitch
		voiced = Get value at time... 'ptime' Hertz Nearest
		if pint > (maxint-threshold) and voiced <> undefined
			select temp_rises
			rindex = Get low index... 'ptime'
			if rindex >= 1
				rtime = Get time from index... 'rindex'
				otime = ('rtime'+'ptime')/2
				otime_'cnt' = otime
				otime2 = otime + 0.05
				otime2_'cnt' = otime2
				cnt += 1
			endif
		endif
	endfor
	select int_tmp
	plus temp_rises
	plus temp_peaks
	plus pitch
	Remove
	for ifile from 1 to cnt-1
		otime = otime_'ifile'
		otime2 = otime2_'ifile'
		select wrk
		ids'ifile' = Extract part... 'otime' 'otime2' Hanning 1 no
	endfor
	if cnt > 1
		select ids1
		for ifile from 2 to cnt-1
			plus ids'ifile'
		endfor
		vowels = Concatenate
		Rename... 's$'_vowels
		select ids1
		for ifile from 2 to cnt-1
			plus ids'ifile'
		endfor
		plus wrk
		Remove
	else
		select wrk
		Remove
		select s
		vowels = Copy... 's$'_vowels
	endif
	select vowels
endproc
