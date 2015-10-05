s1tmp = selected("Sound",1)
s1$ = selected$("Sound",1)
s2tmp = selected("Sound",2)
s2$ = selected$("Sound",2)
select s1tmp
execute workpre.praat
s1 = selected("Sound")
select s2tmp
execute workpre.praat
s2 = selected("Sound")
select s1
include minmaxf0.praat
minF0_1 = minF0
maxF0_1 = maxF0
pitch_1 = To Pitch... 0.01 minF0_1 maxF0_1
pitchtier_1 = Down to PitchTier
select s2
dur = Get total duration
include minmaxf0.praat
minF0_2 = minF0
maxF0_2 = maxF0
manipulation = To Manipulation... 0.01 minF0_2 maxF0_2
plus pitchtier_1
Replace pitch tier
durationtier = Create DurationTier... 's2$' 0 dur
Add point... 0 1
plus manipulation
Replace duration tier
select manipulation
resynthesis = Get resynthesis (overlap-add)
execute workpost.praat
result = Rename... 's2$'_copypitchcontour_'s1$'
select s1
plus s2
plus pitch_1
plus pitchtier_1
plus manipulation
plus durationtier
plus resynthesis
Remove
select result
