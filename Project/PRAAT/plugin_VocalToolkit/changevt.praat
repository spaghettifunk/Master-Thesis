form Change vocal tract
	positive Formant_shift_(0.01-1.99) 1.5
endform
shift = 'formant_shift:2'
if shift < 0.01
	shift = 0.01
endif
if shift > 1.99
	shift = 1.99
endif
include batch.praat
procedure action
	s$ = selected$ ("Sound")
	if shift > 1
		execute workpre.praat
		wrk = selected("Sound")
		dur = Get total duration
		vtfactor = shift
include minmaxf0.praat
		pitch = To Pitch... 0.01 minF0 maxF0
		plus wrk
		manipulation = To Manipulation
		select pitch
		Formula... self/'vtfactor'
		pitchtier = Down to PitchTier
		plus manipulation
		Replace pitch tier
		select pitch
		plus pitchtier
		Remove
		durationtier = Create DurationTier... 's$' 0 dur
		Add point... 0 'vtfactor'
		plus manipulation
		Replace duration tier
		select durationtier
		Remove
		select manipulation
		res = Get resynthesis (overlap-add)
		fsamp = Get sample rate
		tmp = Resample... fsamp/vtfactor 10
		Override sample rate... fsamp
		execute workpost.praat
		result = Rename... 's$'_changevt_'shift'
		select wrk
		plus res
		plus manipulation
		plus tmp
		Remove
		select result
	elsif shift < 1
		execute workpre.praat
		wrk = selected("Sound")
		dur = Get total duration
		vtfactor = 1-'shift'+1
include minmaxf0.praat
		pitch = To Pitch... 0.01 minF0 maxF0+200
		plus wrk
		manipulation = To Manipulation
		select pitch
		Formula... self*vtfactor
		pitchtier = Down to PitchTier
		plus manipulation
		Replace pitch tier
		select pitch
		plus pitchtier
		Remove
		durationtier = Create DurationTier... 's$' 0 dur
		Add point... 0 1/vtfactor
		plus manipulation
		Replace duration tier
		select durationtier
		Remove
		select manipulation
		res = Get resynthesis (overlap-add)
		fsamp = Get sample rate
		tmp = Resample... fsamp*vtfactor 10
		Override sample rate... fsamp
		execute workpost.praat
		result = Rename... 's$'_changevt_'shift'
		select wrk
		plus res
		plus manipulation
		plus tmp
		Remove
		select result
	elsif shift = 1
		result = Copy... 's$'_changevt_1
	endif
endproc
