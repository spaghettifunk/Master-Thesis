form Text to Speech (eSpeak)
	boolean Prelisten_(press_Apply) 1
	optionmenu Language 17
		option Afrikaans
		option Akan-test
		option Albanian
		option Amharic-test
		option Armenian
		option Armenian-west
		option Azerbaijani-test
		option Bosnian
		option Brazil
		option Bulgarian-test
		option Cantonese
		option Catalan
		option Croatian
		option Czech
		option Danish
		option Dari-test
		option Default
		option Divehi-test
		option Dutch-test
		option En-scottish
		option En-westindies
		option English
		option English-us
		option English_rp
		option English_wmids
		option Esperanto
		option Estonian
		option Finnish
		option French
		option French (Belgium)
		option Georgian-test
		option German
		option Greek
		option Greek-ancient
		option Haitian
		option Hindi
		option Hungarian
		option Icelandic-test
		option Indonesian-test
		option Italian
		option Kannada
		option Kazakh
		option Kinyarwanda-test
		option Kurdish
		option Lancashire
		option Latin
		option Latvian
		option Lojban
		option Macedonian-test
		option Malayalam
		option Maltese-test
		option Mandarin
		option Nahuatl - classical
		option Nepali-test
		option Northern-sotho
		option Norwegian
		option Papiamento-test
		option Polish
		option Portugal
		option Punjabi-test
		option Romanian
		option Russian_test
		option Serbian
		option Setswana-test
		option Sinhala
		option Slovak
		option Slovenian-test
		option Spanish
		option Spanish-latin-american
		option Swahili-test
		option Swedish
		option Tamil
		option Telugu
		option Turkish
		option Urdu-test
		option Vietnam
		option Welsh-test
		option Wolof-test
	optionmenu Voice 1
		option default
		option croak
		option f1
		option f2
		option f3
		option f4
		option f5
		option klatt
		option klatt2
		option klatt3
		option m1
		option m2
		option m3
		option m4
		option m5
		option m6
		option m7
		option whisper
		option whisperf
	positive Sampling_frequency_(Hz) 44100
	real Gap_between_words_(s) 0.01
	real Pitch_adjustment_(0-99) 50
	real Pitch_range_(0-99) 50
	natural Words_per_minute_(80-450) 175
	comment Text
	text str 1 2 3 4 5
endform
ss = Create SpeechSynthesizer... "'language$'" 'voice$'
Set speech output settings... 'sampling_frequency' 'gap_between_words' 'pitch_adjustment' 'pitch_range' 'words_per_minute' yes IPA
if prelisten
	Play text... 'str$'
	Remove
else
	To Sound... "'str$'" no
	s$ = selected$("Sound")
	result = Rename... tts_'s$'
	select ss
	Remove
	select result
endif
