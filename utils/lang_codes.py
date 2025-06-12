# CRLs of interest:
# hrln2crls = {
#         "hin": ["hne_Deva", "bho_Deva", "mag_Deva", "mai_Deva", "hin_Deva"],
#         "tur": ["tur_Latn", "uzn_Latn", "tuk_Latn", "azj_Latn", "crh_Latn"],
#         "ita": ["spa_Latn", "fra_Latn", "por_Latn", "ita_Latn", "ron_Latn", "glg_Latn", "cat_Latn", "oci_Latn", "ast_Latn", "lmo_Latn", "vec_Latn", "scn_Latn", "srd_Latn", "fur_Latn", "lij_Latn"],
#         "ind": ["ind_Latn", "jav_Latn", "sun_Latn", "smo_Latn", "mri_Latn", "ceb_Latn", "zsm_Latn", "tgl_Latn", "ilo_Latn", "fij_Latn", "plt_Latn", "pag_Latn"],
#         "arb": ["arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab"]
#         }
    


def flores_code_to_langname(code):
    iso3_to_lang = {
        "eng_Latn": "English",
        "hne_Deva": "Chhattisgarhi",
        "bho_Deva": "Bhojpuri",
        "mag_Deva": "Magahi",
        "mai_Deva": "Maithili",
        "hin_Deva": "Hindi",
        "tam_Taml": "Tamil",
        "tel_Telu": "Telugu",
        "kan_Knda": "Kannada",
        "mal_Mlym": "Malayalam",
        "tur_Latn": "Turkish",
        "uzn_Latn": "Uzbek",
        "tuk_Latn": "Turkmen",
        "azj_Latn": "Azerbaijani",
        "crh_Latn": "Crimean Tatar",
        "spa_Latn": "Spanish",
        "fra_Latn": "French",
        "por_Latn": "Portuguese",
        "ita_Latn": "Italian",
        "ron_Latn": "Romanian",
        "glg_Latn": "Galician",
        "cat_Latn": "Catalan",
        "oci_Latn": "Occitan",
        "ast_Latn": "Asturian",
        "lmo_Latn": "Lombard",
        "vec_Latn": "Venetian",
        "scn_Latn": "Sicilian",
        "srd_Latn": "Sardinian",
        "fur_Latn": "Friulian",
        "lij_Latn": "Ligurian",
        "ind_Latn": "Indonesian",
        "jav_Latn": "Javanese",
        "sun_Latn": "Sundanese",
        "smo_Latn": "Samoan",
        "swh_Latn": "Swahili",
        "mri_Latn": "Maori",
        "mar_Deva": "Marathi",
        "ceb_Latn": "Cebuano",
        "zsm_Latn": "Malay",
        "tgl_Latn": "Tagalog",
        "ilo_Latn": "Ilokano",
        "fij_Latn": "Fijian",
        "plt_Latn": "Plateau Malagasy",
        "pag_Latn": "Pangasinan",
        "arb_Arab": "Arabic",
        "acm_Arab": "Iraqi Arabic",
        "acq_Arab": "Ta'izzi-Adeni Arabic",
        "aeb_Arab": "Tunisian Arabic",
        "ajp_Arab": "South Levantine Arabic",
        "apc_Arab": "North Levantine Arabic",
        "ars_Arab": "Najdi Arabic",
        "ary_Arab": "Moroccan Arabic",
        "arz_Arab": "Egyptian Arabic"
    }
    return iso3_to_lang[code]


def flores_code_to_hrln(code):
    hrln2crls = {
        "hin_Deva": ["hne_Deva", "bho_Deva", "mag_Deva", "mai_Deva", "hin_Deva"],
        "tur_Latn": ["tur_Latn", "uzn_Latn", "tuk_Latn", "azj_Latn", "crh_Latn"],
        "ita_Latn": ["spa_Latn", "fra_Latn", "por_Latn", "ita_Latn", "ron_Latn", "glg_Latn", "cat_Latn", "oci_Latn", "ast_Latn", "lmo_Latn", "vec_Latn", "scn_Latn", "srd_Latn", "fur_Latn", "lij_Latn"],
        "ind_Latn": ["ind_Latn", "jav_Latn", "sun_Latn", "smo_Latn", "mri_Latn", "ceb_Latn", "zsm_Latn", "tgl_Latn", "ilo_Latn", "fij_Latn", "plt_Latn", "pag_Latn"],
        "arb_Arab": ["arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab", \
                     "cai", "dam", "doh", "fes", "jer", "kha", "msa", "riy", "san", "tri", "tun"] # FloRes codes, MADAR codes
        }

    for hrln, crls in hrln2crls.items():
        if code in crls:
            return hrln, flores_code_to_langname(hrln)


def get_crls(hrln):
    
    hrln2crls = {
        "hin": ["hne_Deva", "bho_Deva", "mag_Deva", "mai_Deva", "hin_Deva"],
        "tur": ["tur_Latn", "uzn_Latn", "tuk_Latn", "azj_Latn", "crh_Latn"],
        "ita": ["spa_Latn", "fra_Latn", "por_Latn", "ita_Latn", "ron_Latn", "glg_Latn", "cat_Latn", "oci_Latn", "ast_Latn", "lmo_Latn", "vec_Latn", "scn_Latn", "srd_Latn", "fur_Latn", "lij_Latn"],
        "ind": ["ind_Latn", "jav_Latn", "sun_Latn", "smo_Latn", "mri_Latn", "ceb_Latn", "zsm_Latn", "tgl_Latn", "ilo_Latn", "fij_Latn", "plt_Latn", "pag_Latn"],
        "arb": ["arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab"],
        "arb_madar": ["cai", "dam", "doh", "fes", "jer", "kha", "msa", "riy", "san", "tri", "tun"],
        "hat": ["gcf", "mart1259", "acf", "gcr", "lou", "mfe", "rcf", "crs", "hat"]
        }
    
    return hrln2crls[hrln]

def isocode_to_nllbcode(isocode):
    iso3_to_nllbcode = {
                "hin": "hin_Deva",
                "tur": "tur_Latn",
                "arb": "arb_Arab",
                "rus": "rus_Cyrl",
                "ind": "ind_Latn",
                "ita": "ita_Latn",
                "hat": "hat_Latn",
            }
    return iso3_to_nllbcode[isocode]

def flores_code_to_xlsumname(code):
    '''Conversion from FloRes+ codes to language names in XLSum dataset
    https://huggingface.co/datasets/openlanguagedata/flores_plus
    This is *only* for the purpose of organizing experiments across tasks/datasets.
    Some of these flores-like codes don't exist in FloRes (for languages in XLSum but not in FloRes), they're made up for consistency. 
    '''

    flores2xlsum = {
        "amh_Ethi": "amharic",
        "arb_Arab": "arabic",
        "azb_Latn": "azerbaijani", # azb is the code for South Azerbaijani in flores, there's also North Azerbaijani
        "ben_Beng": "bengali",
        "mya_Mymr": "burmese",
        "cmn_Hans": "chinese_simplified",
        "cmn_Hant": "chinese_traditional",
        "eng_Latn": "english",
        "fra_Latn": "french",
        "guj_Gujr": "gujarati",
        "hau_Latn": "hausa",
        "hin_Deva": "hindi",
        "ibo_Latn": "igbo",
        "ind_Latn": "indonesian",
        "jpn_Jpan": "japanese",
        "run_Latn": "kirundi",
        "kor_Hang": "korean",
        "kir_Cyrl": "kyrgyz",
        "mar_Deva": "marathi",
        "nep_Deva": "nepali",
        "gaz_Latn": "oromo", # gaz is the code for West Central Oromo in flores, no other Oromo in flores
        "pbt_Arab": "pashto", # pbt is the code for Southern Pashto in flores, no other Pashto in flores
        "pes_Arab": "persian", # pes is the code for Western Persian in flores, no other Persian in flores
        "pid_Latn": "pidgin", # West African Pidgin English, not in flores
        "por_Latn": "portuguese",
        "pan_Guru": "punjabi", # pan is the code for Eastern Punjabi in flores, no other Punjabi in flores
        "rus_Cyrl": "russian",
        "gla_Latn": "scottish_gaelic",
        "srp_Cyrl": "serbian_cyrillic", 
        "srp_Latn": "serbian_latin", # Not in flores
        "sin_Sinh": "sinhala",
        "som_Latn": "somali",
        "spa_Latn": "spanish",
        "swa_Latn": "swahili",
        "tam_Taml": "tamil",
        "tel_Telu": "telugu",
        "tha_Thai": "thai",
        "tir_Ethi": "tigrinya",
        "tur_Latn": "turkish",
        "ukr_Cyrl": "ukrainian",
        "urd_Arab": "urdu",
        "uzn_Latn": "uzbek",
        "vie_Latn": "vietnamese",
        "cym_Latn": "welsh",
        "yor_Latn": "yoruba"
    }
    return flores2xlsum[code]


LANGS = {
    # ATTENTION: some of these codes are wrong. Please map to correct ones with map_ours_to_flores
    "eng_Latn": {
        "name": "English",
        "flores_code": "eng_Latn",
        "xlsum_code": "english",
        "gtrans_code": "en",
        "family": "germanic",
        "hrln": "eng_Latn",
    },
    "spa_Latn": {
        "name": "Spanish",
        "flores_code": "spa_Latn",
        "xlsum_code": "spanish",
        "gtrans_code": "es",
        "family": "romance",
        "hrln": "spa_Latn",
    },
    "fra_Latn": {
        "name": "French",
        "flores_code": "fra_Latn",
        "xlsum_code": "french",
        "gtrans_code": "fr",
        "family": "romance",
        "hrln": "fra_Latn",
    },
    "cmn_Hant": {
        "name": "Chinese (Traditional)",
        "flores_code": "cmn_Hant",
        "xlsum_code": "chinese_traditional",
        "gtrans_code": "zh-TW",
        "family": "sino-tibetan",
        "hrln": "cmn_Hant",
    },
    "hin_Deva": {
        "name": "Hindi",
        "flores_code": "hin_Deva",
        "xlsum_code": "hindi",
        "gtrans_code": "hi",
        "family": "indo-aryan",
        "hrln": "hin_Deva",
    },
    "nep_Deva": {
        "name": "Nepali",
        "flores_code": "nep_Deva",
        "xlsum_code": "nepali",
        "gtrans_code": "ne",
        "family": "indo-aryan",
        "hrln": "nep_Deva",
    },
    "ind_Latn": {
        "name": "Indonesian",
        "flores_code": "ind_Latn",
        "xlsum_code": "indonesian",
        "gtrans_code": "id",
        "family": "austronesian",
        "hrln": "ind_Latn",
    },
    "swa_Latn": {
        "name": "Swahili",
        "flores_code": "swa_Latn",
        "xlsum_code": "swahili",
        "gtrans_code": "sw",
        "family": "niger-congo",
        "hrln": "swa_Latn",
    },
    "tur_Latn": {
        "name": "Turkish",
        "flores_code": "tur_Latn",
        "xlsum_code": "turkish",
        "gtrans_code": "tr",
        "family": "turkic",
        "hrln": "tur_Latn",
    },
    "pol_Latn": {
        "name": "Polish",
        "flores_code": "pol_Latn",
        "xlsum_code": "polish",
        "gtrans_code": "pl",
        "family": "slavic",
        "hrln": "pol_Latn",
    },
    "mar_Deva": {
        "name": "Marathi",
        "flores_code": "mar_Deva",
        "xlsum_code": "marathi",
        "gtrans_code": "mr",
        "family": "indo-aryan",
        "hrln": "hin_Deva",
    },
    "ell_Latn": {
        "name": "Greek",
        "flores_code": "ell_Latn",
        "xlsum_code": "greek",
        "gtrans_code": "el",
        "family": "indo-european",
        "hrln": "ell_Latn",
    },
    "kor_Hang": {
        "name": "Korean",
        "flores_code": "kor_Hang",
        "xlsum_code": "korean",
        "gtrans_code": "ko",
        "family": "altaic",
        "hrln": "kor_Hang",
    },
    # "mag_Deva": {
    #     "name": "Magahi",
    #     "flores_code": "mag_Deva",
    #     "xlsum_code": None,
    #     "gtrans_code": "mag",
    #     "family": "indo-aryan",
    #     "hrln": "hin_Deva",
    # },
    "ceb_Latn": {
        "name": "Cebuano",
        "flores_code": "ceb_Latn",
        "xlsum_code": "cebuano",
        "gtrans_code": "ceb",
        "family": "austronesian",
        "hrln": "ind_Latn",
    },
    "cat_Latn": {
        "name": "Catalan",
        "flores_code": "cat_Latn",
        "xlsum_code": None,
        "gtrans_code": "ca",
        "family": "romance",
        "hrln": "cat_Latn",
    },
    "bos_Latn": {
        "name": "Bosnian",
        "flores_code": "bos_Latn",
        "xlsum_code": None,
        "gtrans_code": "bs",
        "family": "slavic",
        "hrln": "bos_Latn",
    },
    "amh_Ethi": {
        "name": "Amharic",
        "flores_code": "amh_Ethi",
        "xlsum_code": "amharic",
        "gtrans_code": "am",
        "family": "semitic",
        "hrln": "amh_Ethi",
    },
    "yor_Latn": {
        "name": "Yoruba",
        "flores_code": "yor_Latn",
        "xlsum_code": "yoruba",
        "gtrans_code": "yo",
        "family": "niger-congo",
        "hrln": "yor_Latn",
    },
    "uzn_Latn": {
        "name": "Uzbek",
        "flores_code": "uzn_Latn",
        "xlsum_code": "uzbek",
        "gtrans_code": "uz",
        "family": "turkic",
        "hrln": "tur_Latn",
    },
    # "yue_Hant": {
    #     "name": "Cantonese",
    #     "flores_code": "yue_Hant",
    #     "xlsum_code": None,
    #     "gtrans_code": "yue",
    #     "family": "sino-tibetan",
    #     "hrln": "yue_Hant",
    # },
    "tam_Taml": {
        "name": "Tamil",
        "flores_code": "tam_Taml",
        "xlsum_code": "tamil",
        "gtrans_code": "ta",
        "family": "dravidian",
        "hrln": "tam_Taml",
    },
    "tel_Telu": {
        "name": "Telugu",
        "flores_code": "tel_Telu",
        "xlsum_code": "telugu",
        "gtrans_code": "te",
        "family": "dravidian",
        "hrln": "tel_Telu",
    },
    # The following languages are plausible intermediate languages, but not source or target languages of interest for our experiments.
    # # Arabic, Chinese (simplified & traditional), Czech, Dutch, English, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, and Vietnamese.
    "cze_Latn": {
        "name": "Czech",
        "flores_code": "cze_Latn",
        "xlsum_code": None,
        "gtrans_code": "cs",
        "family": "slavic",
        "hrln": "cze_Latn",
    },
    "dut_Latn": {
        "name": "Dutch",
        "flores_code": "dut_Latn",
        "xlsum_code": None,
        "gtrans_code": "nl",
        "family": "germanic",
        "hrln": "dut_Latn",
    },
    "heb_Hebr": {
        "name": "Hebrew",
        "flores_code": "heb_Hebr",
        "xlsum_code": None,
        "gtrans_code": "he",
        "family": "semitic",
        "hrln": "heb_Hebr",
    },
    "ita_Latn": {
        "name": "Italian",
        "flores_code": "ita_Latn",
        "xlsum_code": "italian",
        "gtrans_code": "it",
        "family": "romance",
        "hrln": "ita_Latn",
    },
    "far_Arab": {
        "name": "Persian",
        "flores_code": "far_Arab",
        "xlsum_code": None,
        "gtrans_code": "fa",
        "family": "indo-iranian",
        "hrln": "far_Arab",
    },
    "ukr_Cyrl": {
        "name": "Ukrainian",
        "flores_code": "ukr_Cyrl",
        "xlsum_code": None,
        "gtrans_code": "uk",
        "family": "slavic",
        "hrln": "ukr_Cyrl",
    },
    "por_Latn": {
        "name": "Portuguese",
        "flores_code": "por_Latn",
        "xlsum_code": "portuguese",
        "gtrans_code": "pt",
        "family": "romance",
        "hrln": "por_Latn",
    },
    "ron_Latn": {
        "name": "Romanian",
        "flores_code": "ron_Latn",
        "xlsum_code": None,
        "gtrans_code": "ro",
        "family": "romance",
        "hrln": "ron_Latn",
    },
    "rus_Cyrl": {
        "name": "Russian",
        "flores_code": "rus_Cyrl",
        "xlsum_code": "russian",
        "gtrans_code": "ru",
        "family": "slavic",
        "hrln": "rus_Cyrl",
    },
    "vie_Latn": {
        "name": "Vietnamese",
        "flores_code": "vie_Latn",
        "gtrans_code": "vi",
        "family": "austroasiatic",
        "hrln": "vie_Latn",
    },
    "jpn_Jpan": {
        "name": "Japanese",
        "flores_code": "jpn_Jpan",
        "gtrans_code": "ja",
        "family": "altaic",
        "hrln": "jpn_Jpan",
    },
    "cmn_Hans": {
        "name": "Chinese (Simplified)",
        "flores_code": "cmn_Hans",
        "xlsum_code": "chinese_simplified",
        "gtrans_code": "zh-CN",
        "family": "sino-tibetan",
        "hrln": "cmn_Hans",
    },
    "deu_Latn": {
        "name": "German",
        "flores_code": "deu_Latn",
        "gtrans_code": "de",
        "family": "germanic",
        "hrln": "deu_Latn",
    },
    "ara_Arab": {
        "name": "Arabic",
        "flores_code": "ara_Arab",
        "gtrans_code": "ar",
        "family": "semitic",
        "hrln": "ara_Arab",
    },
    "tha_Thai": {
        "name": "Thai",
        "flores_code": "tha_Thai",
        "gtrans_code": "th",
        "family": "tai-kadai",
        "hrln": "tha_Thai",
    },

}

def code_to_langname(code):
    if code in LANGS:
        return LANGS[code]["name"]
    else:
        raise ValueError(f"Language code '{code}' not found in LANGS dictionary.")

def map_flores_to_ours(code):
    flores_to_ours = {
        "ces_Latn": "cze_Latn",
        "ell_Grek": "ell_Latn",
        "nld_Latn": "dut_Latn",
        "arb_Arab": "ara_Arab",
        "pes_Arab": "far_Arab",
        "zho_Hans": "cmn_Hans",
        "zho_Hant": "cmn_Hant",
        "swh_Latn": "swa_Latn",
    }
    if code in flores_to_ours:
        return flores_to_ours[code]
    return code

def map_ours_to_flores(code):
    flores_to_ours = {
        "ces_Latn": "cze_Latn",
        "ell_Grek": "ell_Latn",
        "nld_Latn": "dut_Latn",
        "arb_Arab": "ara_Arab",
        "pes_Arab": "far_Arab",
        "zho_Hans": "cmn_Hans",
        "zho_Hant": "cmn_Hant",
        "swh_Latn": "swa_Latn",
    }
    ours_to_flores = {v: k for k, v in flores_to_ours.items()}
    if code in ours_to_flores:
        return ours_to_flores[code]
    return code


def lang_order(model_name):
    """We want to order languages like this:
    Supported languages, in order of HRL->LRL,
    then unsupported language, in order of HRL->LRL
    ATTENTION: this have correct flores codes! use map_flores_to_ours to map to our codes as in LANGS.
    """
    all_langs = [
        "eng_Latn",
        "spa_Latn",
        "fra_Latn",
        "deu_Latn",
        "rus_Cyrl",
        "jpn_Jpan",
        "por_Latn",
        "ita_Latn",
        "tur_Latn",
        "kor_Hang",
        "arb_Arab",
        "pol_Latn",
        "hin_Deva",
        "heb_Hebr",
        "ukr_Cyrl",
        "zho_Hans",
        "zho_Hant",
        "vie_Latn",
        "ind_Latn",
        "ron_Latn",
        "nld_Latn",
        "ces_Latn",
        "ell_Grek",
        "mar_Deva",
        "swh_Latn",
        "nep_Deva",
        "tam_Taml",
        "tel_Telu",
        "uzn_Latn",
        "cat_Latn",
        "bos_Latn",
        "ceb_Latn",
        "pes_Arab",
        "tha_Thai",
        "amh_Ethi",
        "yor_Latn"
    ]

    supported_langs = {
        "aya": [
            "eng_Latn",   # English
            "spa_Latn",   # Spanish
            "fra_Latn",   # French
            "deu_Latn",   # German
            "rus_Cyrl",   # Russian
            "jpn_Jpan",   # Japanese
            "kor_Hang",   # Korean
            "por_Latn",   # Portuguese
            "ita_Latn",   # Italian
            "zho_Hans",   # Chinese (Simplified)
            "zho_Hant",   # Chinese (Traditional)
            "tur_Latn",   # Turkish
            "hin_Deva",   # Hindi
            "nld_Latn",   # Dutch
            "arb_Arab",   # Arabic
            "pol_Latn",   # Polish
            "ukr_Cyrl",   # Ukrainian
            "heb_Hebr",   # Hebrew
            "ces_Latn",   # Czech
            "ron_Latn",   # Romanian
            "ind_Latn",   # Indonesian
            "vie_Latn",   # Vietnamese
            "ell_Grek",   # Greek
            "pes_Arab",   # Persian
        ],
        "llama": [
            "eng_Latn",   # English
            "spa_Latn",   # Spanish
            "fra_Latn",   # French
            "deu_Latn",   # German
            "ita_Latn",   # Italian
            "por_Latn",   # Portuguese
            "hin_Deva",   # Hindi
            "tha_Thai",   # Thai
        ]
    }
    if "aya" in model_name:
        supported_langs = supported_langs["aya"]
    elif "llama" in model_name:
        supported_langs = supported_langs["llama"]

    unsupported_langs = [lang for lang in all_langs if lang not in supported_langs]
    return supported_langs, unsupported_langs
    

def label_map(lang_code, supported=None, unsupported=None, exp_key=None):
    # First convert to flores
    if exp_key is not None:
        supported, unsupported = lang_order(exp_key)
    lang_code = map_ours_to_flores(lang_code)
    if "zho" in lang_code:
        lang = f"{lang_code.split('_')[0]}({lang_code[-1]})"
    else:
        lang = lang_code.split("_")[0]
    if lang_code in supported:
        lang += "*"
    return lang
