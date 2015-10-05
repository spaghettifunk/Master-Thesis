smoothing = 60
semitones_stylize = 0.5
optimizeTime = 0.12
semitones_diff = 0.7
unvoiced_part_threshold = 0.3
include batch.praat
procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	sample_rate = Get sample rate
	duration = Get total duration
	result = Copy... 's$'_markbypitch
	execute fixdc.praat
	intensity = To Intensity... 40 0.01 1
	select result
include minmaxf0.praat
	pitch_tmp = To Pitch (ac)... 0.01 minF0 15 no 0.05 0.45 0.01 0.35 0.16 maxF0
	pitch = Smooth... smoothing
	pitchtier = Down to PitchTier
 	Stylize... semitones_stylize Semitones
	select pitch
	pointprocess = To PointProcess
	textgrid = To TextGrid (vuv)... 0.02 0.01
	Rename... 's$'_markbypitch
	plus s
	select textgrid
	call MarkSemitones
	call OptimizeTimes
	call OptimizeTimes
	call OptimizeSemitones
	call JoinUnvoiced
	call SplitEqualNotes
	call OptimizeTimes
	call ZeroCrossing
	select pitch_tmp
	plus pitch
	plus pointprocess
	plus pitchtier
	plus intensity
	Remove
	select textgrid
	plus result
	Edit
endproc
procedure MarkSemitones
	select textgrid
	n_intervals = Get number of intervals... 1
	for i from 1 to n_intervals
		interval_label$ = Get label of interval... 1 i
		if interval_label$ = "V"
			Set interval text... 1 i
			starting_point =	Get starting point... 1 i
			end_point =  Get end point... 1 i
			select pitchtier
			starting_index = Get nearest index from time... starting_point
			end_index = Get nearest index from time... end_point
			for t from starting_index to end_index
				index_time_'t' = Get time from index... t
			endfor
			select textgrid
			st_ind = starting_index+1
			en_ind = end_index-1
			for ii from st_ind to en_ind
				ind_tim = index_time_'ii'
				Insert boundary... 1 ind_tim
				n_intervals += 1
				i += 1
			endfor
		endif
	endfor
endproc
procedure OptimizeTimes
	select intensity
	t = 0.01
	n = round(duration*100)
	for i from 1 to n
		intensity_value_'i' = Get value at time... t Nearest
		intensity_value_'i' = floor(intensity_value_'i')
		t = t + 0.01
	endfor
	select textgrid
	i = Get number of intervals... 1
	while i>1
		current_label$ = Get label of interval... 1 i
		prev_label$ = Get label of interval... 1 i-1
		n_intervals = Get number of intervals... 1
		if i < n_intervals
			next_label$ = Get label of interval... 1 i+1
		else
			next_label$ = "U"
		endif
		starting_point = Get starting point... 1 i
		end_point = Get end point... 1 i
		interval_time = end_point-starting_point
		interval_time = 'interval_time:2'
		starting_index = round('starting_point'*100)
		if starting_index = 0
			starting_index = 1
		endif
		end_index = round('end_point'*100)
		starting_intensity_value = intensity_value_'starting_index'
		end_intensity_value = intensity_value_'end_index'
		if starting_intensity_value = undefined
			starting_intensity_value = 0
		endif
		if end_intensity_value = undefined
			end_intensity_value = 0
		endif
		if current_label$ <> "U" and prev_label$ <> "U" and next_label$ <> "U" and interval_time < optimizeTime
			select intensity
			minimum_time = Get time of minimum... starting_point end_point Parabolic
			select textgrid
			Remove boundary at time... 1 starting_point
			Remove boundary at time... 1 end_point
			Insert boundary... 1 minimum_time
		endif
		if current_label$ <> "U" and prev_label$ <> "U" and next_label$ = "U" and interval_time < optimizeTime
			Remove boundary at time... 1 starting_point
		endif
		if current_label$ <> "U" and prev_label$ = "U" and next_label$ <> "U" and interval_time < optimizeTime
			Remove boundary at time... 1 end_point
		endif
		if current_label$ <> "U" and prev_label$ = "U" and next_label$ = "U" and interval_time < 0.075
			Set interval text... 1 i-1
			Remove boundary at time... 1 starting_point
			Remove boundary at time... 1 end_point
		endif
		if current_label$ = "U" and interval_time < 0.075 and i < n_intervals
			select intensity
			minimum_time = Get time of minimum... starting_point end_point Parabolic
			select textgrid
			Set interval text... 1 i
			Remove boundary at time... 1 starting_point
			Remove boundary at time... 1 end_point
			Insert boundary... 1 minimum_time
		endif
		i -= 1
	endwhile
endproc
procedure OptimizeSemitones
	call GetPitchValues
	select textgrid
	i = n_intervals
	while i>1
		current_label$ = Get label of interval... 1 i
		prev_label$ = Get label of interval... 1 i-1
		if current_label$ <> "U"
			current_interval_semitone_first = interval_semitone_first_'i'
			i_prev = i-1
			prev_interval_semitone = interval_semitone_'i_prev'
			prev_interval_semitone_last = interval_semitone_last_'i_prev'
			if prev_interval_semitone <> undefined and prev_label$ <> "U"
				diff_semitones = abs(current_interval_semitone_first-prev_interval_semitone_last)
				diff_semitones = 'diff_semitones:1'
			else
				diff_semitones = semitones_diff
			endif
			if diff_semitones < semitones_diff
				starting_point = Get starting point... 1 i
				Remove boundary at time... 1 starting_point
				Set interval text... 1 i-1 'prev_label$'
			endif
		endif
		i -= 1
	endwhile
endproc
procedure JoinUnvoiced
	select textgrid
	n_intervals = Get number of intervals... 1
	i = n_intervals-1
	while i>1
		current_label$ = Get label of interval... 1 i
		prev_label$ = Get label of interval... 1 i-1
		next_label$ = Get label of interval... 1 i+1
		starting_point = Get starting point... 1 i
		end_point = Get end point... 1 i
		interval_time = end_point-starting_point
		interval_time = 'interval_time:2'
		if current_label$ = "U" and interval_time <= unvoiced_part_threshold
			mid_point = (starting_point+end_point)/2
			Set interval text... 1 i
			Remove boundary at time... 1 starting_point
			Remove boundary at time... 1 end_point
			Insert boundary... 1 mid_point
			Set interval text... 1 i-1 'prev_label$'
			Set interval text... 1 i 'next_label$'
		endif
		i -= 1
	endwhile
endproc
procedure SplitEqualNotes
	select textgrid
	n_intervals = Get number of intervals... 1
	i = n_intervals-1
	while i>1
		current_label$ = Get label of interval... 1 i
		starting_point = Get starting point... 1 i
		end_point = Get end point... 1 i
		interval_time = end_point-starting_point
		interval_margin = interval_time/3
		interval_margin = 'interval_margin:3'
		if current_label$ <> "U" and interval_time > 0.6
			select intensity
			minimum_time = Get time of minimum... starting_point+interval_margin end_point-interval_margin Parabolic
			intensity_sd = Get standard deviation... starting_point+interval_margin end_point-interval_margin
			select textgrid
			if intensity_sd > 3
				Insert boundary... 1 minimum_time
			endif
		endif
		i -= 1
	endwhile
endproc
procedure GetPitchValues
	select textgrid
	n_intervals = Get number of intervals... 1
	for i from 1 to n_intervals
		starting_point_'i' =	Get starting point... 1 i
		end_point_'i' =  Get end point... 1 i
		interval_label_'i'$ = Get label of interval... 1 i
	endfor
	select pitch
	for i from 1 to n_intervals
		starting_point = starting_point_'i'
		end_point = end_point_'i'
		interval_time_'i' = end_point-starting_point
		interval_margin = interval_time_'i'/4
		interval_margin = 'interval_margin:3'
		interval_semitone_'i' = Get quantile... starting_point end_point 0.5 semitones re 440 Hz
		if interval_time_'i' > 0.17
			interval_semitone_first_'i' = Get quantile... starting_point starting_point+0.17 0.5 semitones re 440 Hz
			interval_semitone_last_'i' = Get quantile... end_point-0.17 end_point 0.5 semitones re 440 Hz
		else
			interval_semitone_first_'i' = Get quantile... starting_point end_point 0.5 semitones re 440 Hz
			interval_semitone_last_'i' = Get quantile... starting_point end_point 0.5 semitones re 440 Hz
		endif
	endfor
endproc
procedure ZeroCrossing
	select textgrid
	n_intervals = Get number of intervals... 1
	for i from 2 to n_intervals
		starting_point_'i' = Get starting point... 1 i
	endfor
	select result
	for i from 2 to n_intervals
		starting_point = starting_point_'i'
		zero_crossing_'i' = Get nearest zero crossing... Left starting_point
	endfor
	select textgrid
	for i from 2 to n_intervals
		starting_point = starting_point_'i'
		zero_crossing = zero_crossing_'i'
		interval_label$ = Get label of interval... 1 i
		Set interval text... 1 i
		Remove boundary at time... 1 starting_point
		Insert boundary... 1 zero_crossing
		Set interval text... 1 i 'interval_label$'
	endfor
endproc
