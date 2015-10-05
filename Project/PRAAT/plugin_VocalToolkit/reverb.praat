form Reverb
	boolean Prelisten_(press_Apply) 1
	positive Prelisten_time_(s) 3.0
	optionmenu Preset 1
		option Ambience
		option Bath
		option Church
		option Hall
		option Plate
		option Robot
		option Room Big
		option Room Medium
		option Room Small
		option Stadium
		option Studio
		option Vocal
	natural Mix_(1-100_%) 50
endform
if mix > 100
	mix = 100
endif
if mix <= 50
	gain1 = 1
	gain2 = mix/50
endif
if mix > 50
	gain1 = (100-mix)/50
	gain2 = 1
endif
include batch.praat
procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	if prelisten
		sordur = Get total duration
		if prelisten_time > sordur
			prelisten_time = sordur
		endif
		pre = Extract part... 0 prelisten_time rectangular 1 no
	endif
	soundname$ = "'s$'_reverb_'preset$'_'mix'"
	reverb_ir = Read from file... reverb/'preset$'.flac
	ch = Get number of channels
	sr1 = 1/Object_'wrk'.dx
	sr2 = 1/Object_'reverb_ir'.dx
	if prelisten
		select pre
	else
		select wrk
	endif
	if sr1<>sr2
		stmp = Resample... sr2 50
	else
		stmp = Copy... tmp
	endif
	int = Get intensity (dB)
	Scale... 0.9999
	plus reverb_ir
	reverb = Convolve
	Scale... 0.9999
	dur = Get total duration
	result = Create Sound from formula... "'soundname$'" 'ch' 0 dur sr2 0
	Formula...  self + 'gain1'*Object_'stmp'[]
	Formula...  self + 'gain2'*Object_'reverb'[]
	Scale intensity... int
	execute declip.praat
	select wrk
	plus reverb_ir
	plus stmp
	plus reverb
	Remove
	select result
	if prelisten
		Play
		plus pre
		Remove
		select s
	endif
endproc
