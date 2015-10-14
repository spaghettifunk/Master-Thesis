s1 = selected("Sound",1)
s1$ = selected$("Sound",1)
s2 = selected("Sound",2)
s2$ = selected$("Sound",2)
select s2

sr2 = Get sample rate
execute workpre.praat
tmp1 = selected("Sound")
select s1
sr1 = Get sample rate
execute workpre.praat
if sr1<>sr2
	tmp3 = selected("Sound")
	Resample... sr2 50
endif
tmp2 = selected("Sound")

select tmp1
plus tmp2
Scale... 0.9999
To MFCC... 12 0.015 0.005 100.0 100.0 0.0
mfcc1 = selected("MFCC", 1)
mfcc2 = selected("MFCC", 2)
dtw = nowarn To DTW...  1.0 0.0 0.0 0.0 0.056 yes yes 2/3 < slope < 3/2
select tmp1
dur = Get total duration
select dtw
curtime = 0
nexttime = 0
ntime = 0
interval = 0
pnts = floor(dur/0.1)
repeat
	if ntime = 0
		curpath = 0
	else
		curpath = Get y time from x time... 'curtime'
	endif
	nexttime = curtime + 0.1
	nextpath = Get y time from x time... 'nexttime'
	interval = nextpath-curpath
	curpath_'ntime' = curpath
	nextpath_'ntime' = nextpath
	interval_'ntime' = interval
	curtime = curtime + 0.1
	ntime += 1
until ntime = pnts
durationtier = Create DurationTier... DTW 0 'dur'
ntime = 0
int_or = 0.1
repeat
	curpath = curpath_'ntime'
	nextpath = nextpath_'ntime'
	interval = interval_'ntime'
	if interval = 0
		int_or = int_or + 0.1
	else
		int_or = 0.1
	endif
	stpnt = curpath + 0.000001
	endpnt = nextpath - 0.000001
	if interval<>0
		Add point... 'curpath' 1
		Add point... 'stpnt' 'int_or'/'interval'
		Add point... 'endpnt' 'int_or'/'interval'
	endif
	ntime += 1
until ntime = pnts
Add point... 'endpnt'+0.000001 1
select tmp1
include minmaxf0.praat
manipulation = To Manipulation... 0.01 minF0 maxF0
plus durationtier
Replace duration tier
select manipulation
resynthesis = Get resynthesis (overlap-add)
execute workpost.praat
result = Rename... 's2$'_dtw_'s1$'
select tmp1
plus tmp2
plus mfcc1
plus mfcc2
plus dtw
plus durationtier
plus manipulation
plus resynthesis
if sr1<>sr2
	plus tmp3
endif
Remove
select result
