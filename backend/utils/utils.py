from tldextract import tldextract


class Utils:

    @staticmethod
    def is_eng_suffix(self, url):
        result = tldextract.extract(url)
        if result.suffix == "com":
            return True
        elif result.suffix == "us":
            return True
        elif result.suffix == "uk":
            return True
        elif result.suffix == "co.uk":
            return True
        elif result.suffix == "au":
            return True
        elif result.suffix == "com.au":
            return True
        return False

    # Not complete
    # Further trims down - easing job of nlp
    @staticmethod
    def is_eng_subdomain(self, url):
        result = tldextract.extract(url)
        debug = result.subdomain
        if "es" in result.suffix:
            return False
        elif "it" in result.suffix:
            return False
        elif "mx" in result.suffix:
            return False
        elif "de" in result.suffix:
            return False
        elif "fr" in result.suffix:
            return False
        elif "kr" in result.suffix:
            return False
        return True