// Shared local text-PII detection. Loaded as a classic script in BOTH the
// content-script world (before content.js) and the popup (before popup.js's
// module), so the page scanner and the network boundary use one pattern set and
// cannot drift apart.
//
// Everything here is pure, local and offline: no network, no model, no OCR.
(() => {
  const REDACTION_PLACEHOLDER = "[REDACTED_TEXT_PII]";

  // Text hosts whose contents are never rendered as visible page text.
  const NON_RENDERED_TAGS = new Set([
    "script",
    "style",
    "noscript",
    "template",
    "head",
    "title",
    "meta",
    "link",
    "iframe",
    "object",
    "svg",
    "canvas"
  ]);

  function passesLuhn(digits) {
    let sum = 0;
    let double = false;

    for (let index = digits.length - 1; index >= 0; index -= 1) {
      let value = digits.charCodeAt(index) - 48;

      if (value < 0 || value > 9) {
        return false;
      }

      if (double) {
        value *= 2;
        if (value > 9) {
          value -= 9;
        }
      }

      sum += value;
      double = !double;
    }

    return sum % 10 === 0;
  }

  // Aadhaar numbers carry a Verhoeff check digit. Validating it turns a bare
  // 12-digit run from a coin flip into a ~10x stronger signal, which is what
  // keeps order numbers and long product codes out of the redaction set.
  const VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  ];
  const VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
  ];

  function passesVerhoeff(digits) {
    let checksum = 0;
    const reversed = digits.split("").reverse();

    for (let index = 0; index < reversed.length; index += 1) {
      const digit = Number(reversed[index]);

      if (!Number.isInteger(digit)) {
        return false;
      }

      checksum = VERHOEFF_D[checksum][VERHOEFF_P[index % 8][digit]];
    }

    return checksum === 0;
  }

  // Ordered most-specific first. `priority` breaks ties when two patterns claim
  // the same span; `validate` is where the false-positive filtering lives.
  const PII_PATTERNS = [
    {
      type: "email",
      priority: 100,
      regex: /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}/g
    },
    {
      type: "sensitive-url-param",
      priority: 95,
      regex: /\b(?:token|api[_-]?key|apikey|access[_-]?token|auth|session(?:[_-]?id)?|password|pwd|passwd|secret|otp|aadhaar|pan|ssn|email|phone|mobile)=[^&\s"'<>]{4,}/gi
    },
    {
      type: "api-key",
      priority: 94,
      regex: /\b(?:sk|gsk|pk|rk)[-_][A-Za-z0-9_-]{16,}\b/g
    },
    {
      // 13-19 digits, optionally grouped. Luhn is what stops this from eating
      // long order numbers and identifiers.
      type: "payment-card",
      priority: 90,
      regex: /\b(?:\d[ -]?){12,18}\d\b/g,
      validate(text) {
        const digits = text.replace(/\D/g, "");
        return digits.length >= 13 && digits.length <= 19 && passesLuhn(digits);
      }
    },
    {
      // Grouped Aadhaar (XXXX XXXX XXXX). The grouping itself is a strong
      // format signal, so this form is accepted without the checksum.
      type: "aadhaar",
      priority: 86,
      regex: /\b[2-9]\d{3}[ -]\d{4}[ -]\d{4}\b/g
    },
    {
      // Bare 12-digit Aadhaar needs the Verhoeff check digit to qualify.
      type: "aadhaar",
      priority: 85,
      regex: /\b[2-9]\d{11}\b/g,
      validate: passesVerhoeff
    },
    {
      type: "ssn",
      priority: 80,
      regex: /\b\d{3}-\d{2}-\d{4}\b/g
    },
    {
      type: "pan",
      priority: 75,
      regex: /\b[A-Z]{5}\d{4}[A-Z]\b/g
    },
    {
      // Any "+countrycode ..." run. Real-world grouping varies too much to
      // pin down (+91 98765 43210 vs +1 (555) 867-5309), so the shape stays
      // loose and the digit-count check does the false-positive filtering.
      type: "phone",
      priority: 72,
      regex: /\+\d[\d .\-()]{6,16}\d/g,
      validate(text) {
        const digits = text.replace(/\D/g, "");
        return digits.length >= 8 && digits.length <= 15;
      }
    },
    {
      type: "phone",
      priority: 71,
      regex: /\b\d{3}[ .-]\d{3}[ .-]\d{4}\b/g
    },
    {
      // Indian mobile numbers are exactly 10 digits starting 6-9. Requiring the
      // leading digit keeps 10-digit order/product IDs out.
      type: "phone",
      priority: 70,
      regex: /\b[6-9]\d{9}\b/g
    },
    {
      type: "ipv4",
      priority: 60,
      regex: /\b(?:\d{1,3}\.){3}\d{1,3}\b/g,
      validate(text) {
        const octets = text.split(".");

        if (!octets.every((octet) => octet.length <= 3 && Number(octet) <= 255)) {
          return false;
        }

        // "1.2.3.4" is far more often a version string than an address.
        return !octets.every((octet) => octet.length === 1);
      }
    },
    {
      type: "ipv6",
      priority: 59,
      regex: /\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b/g
    }
  ];

  // ---------------------------------------------------------------------------
  // Contextual entity rules (feature #6).
  //
  // This is deliberately NOT statistical named-entity recognition. Every rule
  // fires only on an explicit contextual cue: a field label ("Name:",
  // "Address:"), a relationship phrase ("works at", "lives in"), an honorific
  // ("Dr."), a greeting ("Welcome, "), or a strict structural shape (a numbered
  // street address). Bare capitalised words, UI labels, product names and
  // context-free dates/times are left alone on purpose. Deterministic PII
  // (email, phone, card, Aadhaar, PAN, SSN, IP, keys, URL params) stays in
  // PII_PATTERNS above and is not duplicated here.
  // ---------------------------------------------------------------------------

  const ENTITY_PLACEHOLDERS = {
    person: "[REDACTED_PERSON]",
    location: "[REDACTED_LOCATION]",
    address: "[REDACTED_ADDRESS]",
    organization: "[REDACTED_ORGANIZATION]",
    date: "[REDACTED_DATE]",
    time: "[REDACTED_TIME]",
    age: "[REDACTED_AGE]",
    username: "[REDACTED_USERNAME]",
    "job-title": "[REDACTED_JOB_TITLE]",
    "misc-personal": "[REDACTED_PERSONAL]"
  };

  // Entities rank below every deterministic pattern, so an exact tie on the
  // same span keeps the more precise deterministic type.
  const ENTITY_PRIORITY = {
    person: 50,
    address: 48,
    organization: 46,
    username: 44,
    "job-title": 42,
    location: 40,
    "misc-personal": 38,
    date: 36,
    time: 35,
    age: 34
  };

  // Words that are capitalised in ordinary UI copy but are not names,
  // organisations or places. Any hit rejects a person/organisation candidate.
  const STOP_WORDS = new Set([
    "a", "about", "account", "admin", "administrator", "agent", "all", "an", "and", "apple",
    "available", "back", "blog", "call", "cancel", "care", "cart", "center", "centre", "chat",
    "choose", "click", "close", "contact", "continue", "customer", "customers", "dashboard",
    "delete", "department", "desk", "details", "edit", "edition", "email", "enter", "everyone",
    "facebook", "for", "friend", "friends", "galaxy", "github", "google", "guest", "guests",
    "help", "her", "here", "his", "home", "in", "info", "information", "learn", "linkedin",
    "log", "login", "logout", "lite", "max", "me", "member", "members", "menu", "message",
    "microsoft", "mini", "mobile", "more", "my", "name", "new", "news", "next", "none", "not",
    "now", "of", "office", "online", "open", "optional", "or", "order", "orders", "our",
    "out", "page", "password", "phone", "plus", "policy", "previous", "privacy", "pro",
    "profile", "provided", "required", "sales", "save", "search", "select", "service",
    "services", "settings", "shop", "sign", "signin", "sir", "madam", "store", "submit",
    "support", "team", "terms", "that", "the", "their", "there", "this", "to", "today",
    "tomorrow", "twitter", "ultra", "unknown", "up", "us", "user", "username", "users",
    "welcome", "world", "yesterday", "you", "your"
  ]);

  const CORPORATE_SUFFIXES = new Set([
    "technologies", "technology", "tech", "solutions", "systems", "software", "labs", "pvt",
    "private", "ltd", "limited", "inc", "llc", "llp", "corp", "corporation", "company", "co",
    "industries", "enterprises", "group", "holdings", "consulting", "consultants",
    "hospital", "hospitals", "clinic", "university", "college", "institute", "school", "bank",
    "foundation", "trust", "associates", "infotech", "infosystems", "global", "international"
  ]);

  const ORG_ABBREVIATIONS = new Set(["pvt", "ltd", "inc", "co", "corp", "llc", "llp", "bros", "intl"]);

  // Compact gazetteer of Indian cities/states and a few major world cities.
  // A gazetteer hit alone never redacts: it needs a preposition or label.
  const PLACE_NAMES = [
    "Mumbai", "Delhi", "New Delhi", "Kolkata", "Calcutta", "Chennai", "Madras", "Bengaluru",
    "Bangalore", "Hyderabad", "Pune", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur",
    "Nagpur", "Indore", "Bhopal", "Patna", "Ranchi", "Guwahati", "Bhubaneswar", "Chandigarh",
    "Noida", "Gurugram", "Gurgaon", "Ghaziabad", "Faridabad", "Thane", "Navi Mumbai",
    "Vadodara", "Coimbatore", "Kochi", "Cochin", "Thiruvananthapuram", "Trivandrum",
    "Visakhapatnam", "Vijayawada", "Madurai", "Mysuru", "Mysore", "Mangaluru", "Varanasi",
    "Agra", "Prayagraj", "Allahabad", "Amritsar", "Ludhiana", "Dehradun", "Shimla", "Srinagar",
    "Jammu", "Raipur", "Jodhpur", "Udaipur", "Nashik", "Aurangabad", "Rajkot", "Durgapur",
    "Asansol", "Siliguri", "Howrah", "Jamshedpur", "Dhanbad", "Panaji", "West Bengal",
    "Maharashtra", "Karnataka", "Tamil Nadu", "Kerala", "Telangana", "Andhra Pradesh",
    "Uttar Pradesh", "Madhya Pradesh", "Bihar", "Jharkhand", "Odisha", "Rajasthan", "Gujarat",
    "Punjab", "Haryana", "Assam", "Uttarakhand", "Himachal Pradesh", "Chhattisgarh", "Goa",
    "London", "Paris", "Berlin", "Tokyo", "Singapore", "Dubai", "Sydney", "Toronto",
    "New York", "San Francisco", "Los Angeles", "Chicago", "Boston", "Seattle"
  ];
  const PLACE_SET = new Set(PLACE_NAMES);
  const PLACE_ALTERNATION = [...PLACE_NAMES]
    .sort((first, second) => second.length - first.length)
    .map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");

  const isStopWord = (token) => STOP_WORDS.has(token.toLowerCase().replace(/[.'’]+$/, ""));
  const startsWithPlaceholder = (text, index) => text.startsWith("[REDACTED_", index);

  // Read up to `maxTokens` capitalised name tokens separated by single spaces.
  const NAME_TOKEN = /[A-Z][a-z]+(?:[A-Z][a-z]+)?(?:['’-][A-Za-z]+)?|[A-Z]\./y;

  function readNameRun(text, from, maxTokens = 3) {
    const tokens = [];
    let cursor = from;
    let end = from;

    while (tokens.length < maxTokens) {
      NAME_TOKEN.lastIndex = cursor;
      const match = NAME_TOKEN.exec(text);

      if (!match) {
        break;
      }

      const tokenEnd = cursor + match[0].length;

      if (/[A-Za-z0-9_@]/.test(text.charAt(tokenEnd))) {
        break;
      }

      tokens.push(match[0]);
      end = tokenEnd;

      if (text.charAt(tokenEnd) === " " && /[A-Z]/.test(text.charAt(tokenEnd + 1))) {
        cursor = tokenEnd + 1;
      } else {
        break;
      }
    }

    const hasRealName = tokens.some((token) => !token.endsWith("."));
    return tokens.length > 0 && hasRealName ? { start: from, end, tokens } : null;
  }

  // Organisation names: capitalised tokens, lowercase connectors allowed
  // between them, and a trailing "." kept only on known abbreviations.
  const ORG_TOKEN = /[A-Z][A-Za-z0-9&'’-]*/y;
  const ORG_CONNECTOR = /(?:of|and|&|for|the|de) (?=[A-Z])/y;

  function readOrgRun(text, from, maxTokens = 6) {
    const tokens = [];
    let cursor = from;
    let end = from;

    while (tokens.length < maxTokens) {
      ORG_TOKEN.lastIndex = cursor;
      const match = ORG_TOKEN.exec(text);

      if (!match) {
        break;
      }

      let tokenEnd = cursor + match[0].length;
      const token = match[0];

      if (text.charAt(tokenEnd) === "." && ORG_ABBREVIATIONS.has(token.toLowerCase())) {
        tokenEnd += 1;
      }

      tokens.push(token);
      end = tokenEnd;

      if (text.charAt(tokenEnd) !== " ") {
        break;
      }

      if (/[A-Z]/.test(text.charAt(tokenEnd + 1))) {
        cursor = tokenEnd + 1;
        continue;
      }

      ORG_CONNECTOR.lastIndex = tokenEnd + 1;
      const connector = ORG_CONNECTOR.exec(text);

      if (!connector) {
        break;
      }

      cursor = tokenEnd + 1 + connector[0].length;
    }

    if (tokens.length === 0 || isStopWord(tokens[0])) {
      return null;
    }

    return { start: from, end, tokens };
  }

  // Free-form words (job titles, personal attributes), stopping at connectors,
  // punctuation and the next "Label:".
  const WORD_TOKEN = /[A-Za-z][A-Za-z&/+-]*/y;
  const WORD_STOPS = new Set(["at", "in", "for", "with", "from", "and", "who", "since", "he", "she", "they", "is", "was"]);

  function readWordRun(text, from, maxTokens) {
    let cursor = from;
    let end = from;
    let count = 0;

    while (count < maxTokens) {
      WORD_TOKEN.lastIndex = cursor;
      const match = WORD_TOKEN.exec(text);

      if (!match || WORD_STOPS.has(match[0].toLowerCase())) {
        break;
      }

      const tokenEnd = cursor + match[0].length;

      // "Male Age: 21" - the next field's label is not part of this value.
      if (count > 0 && /^\s*:/.test(text.slice(tokenEnd, tokenEnd + 2))) {
        break;
      }

      if (count === 0 && isStopWord(match[0])) {
        return null;
      }

      count += 1;
      end = tokenEnd;

      if (text.charAt(tokenEnd) === " " && /[A-Za-z]/.test(text.charAt(tokenEnd + 1))) {
        cursor = tokenEnd + 1;
      } else {
        break;
      }
    }

    return count > 0 ? { start: from, end } : null;
  }

  function eachTrigger(regex, text, callback) {
    regex.lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      if (match[0].length === 0) {
        regex.lastIndex += 1;
        continue;
      }

      callback(match.index + match[0].length, match);
    }
  }

  // --- PERSON ---------------------------------------------------------------

  const PERSON_LABEL = new RegExp(
    String.raw`(?<![A-Za-z][ \t]?)(?:(?:your|my|full|first|last|middle|legal|contact|customer|client|patient|guest|employee|student|applicant|account\s+holder|card\s*holder|father'?s|mother'?s|spouse'?s|guardian'?s|nominee'?s?|passenger|beneficiary)\s+)?name\s*[:\-]\s*` +
    String.raw`|(?<![A-Za-z][ \t]?)(?:patient|customer|client|passenger|applicant|beneficiary|nominee|account\s+holder|card\s*holder|guardian|spouse|father|mother|emergency\s+contact|contact\s+person|bill\s+to|ship\s+to|recipient|consignee|signed\s+by|prepared\s+for)\s*[:\-]\s*`,
    "gi"
  );
  const PERSON_HONORIFIC = /\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|Shri|Smt|Sri|Kumari)\.?\s+/g;
  const PERSON_GREETING = /\b(?:Hi|Hello|Hey|Dear|Welcome(?:\s+back)?|Namaste|Thanks|Thank\s+you|Congratulations|Congrats|Goodbye|Bye)\s*,?\s+/gi;
  const PERSON_SESSION = /\b(?:signed\s+in\s+as|logged\s+in\s+as|continue\s+as|continuing\s+as|on\s+behalf\s+of|s\/o|d\/o|w\/o|c\/o|son\s+of|daughter\s+of|wife\s+of|husband\s+of)\s+/gi;
  // Weak cues ("Contact", "Posted by") need at least two name tokens.
  const PERSON_WEAK = /\b(?:contact|call|meet|message|assigned\s+to|requested\s+by|submitted\s+by|posted\s+by|written\s+by|authored\s+by|booked\s+by|reviewed\s+by|approved\s+by|prescribed\s+by|referred\s+by|attended\s+by|treated\s+by)\s+/gi;

  function personRules(text, push) {
    const tryRun = (from, minTokens) => {
      const run = readNameRun(text, from, 3);

      if (!run || run.tokens.length < minTokens || run.tokens.some(isStopWord)) {
        return;
      }

      push("person", run.start, run.end);
    };

    eachTrigger(PERSON_LABEL, text, (from) => tryRun(from, 1));
    eachTrigger(PERSON_HONORIFIC, text, (from) => tryRun(from, 1));
    eachTrigger(PERSON_GREETING, text, (from) => tryRun(from, 1));
    eachTrigger(PERSON_SESSION, text, (from) => tryRun(from, 1));
    eachTrigger(PERSON_WEAK, text, (from) => tryRun(from, 2));
  }

  // --- LOCATION -------------------------------------------------------------

  const LOCATION_PREPOSITION = new RegExp(
    String.raw`\b(?:[Ii]n|[Ff]rom|[Nn]ear|[Aa]t|[Rr]esident\s+of|[Nn]ative\s+of)\s+(` + PLACE_ALTERNATION + String.raw`)(?![A-Za-z])`,
    "g"
  );
  const LOCATION_LABEL_STRONG = /(?<![A-Za-z][ \t]?)(?:city|town|village|district|hometown|home\s+town|native\s+place|place\s+of\s+birth|birth\s*place|current\s+city|residence)\s*[:\-]\s*/gi;
  const LOCATION_PHRASE = /\b(?:lives\s+in|living\s+in|based\s+in|resides\s+in|residing\s+in|moved\s+to|relocated\s+to)\s+/gi;
  const LOCATION_LABEL_WEAK = /(?<![A-Za-z][ \t]?)(?:location|state|region|area)\s*[:\-]\s*/gi;

  function locationRules(text, push) {
    LOCATION_PREPOSITION.lastIndex = 0;
    let match;

    while ((match = LOCATION_PREPOSITION.exec(text)) !== null) {
      const end = match.index + match[0].length;
      push("location", end - match[1].length, end);
    }

    const strong = (from) => {
      const run = readNameRun(text, from, 3);

      if (run && !run.tokens.some(isStopWord)) {
        push("location", run.start, run.end);
      }
    };

    eachTrigger(LOCATION_LABEL_STRONG, text, strong);
    eachTrigger(LOCATION_PHRASE, text, strong);
    eachTrigger(LOCATION_LABEL_WEAK, text, (from) => {
      const run = readNameRun(text, from, 3);

      if (run && PLACE_SET.has(run.tokens.join(" "))) {
        push("location", run.start, run.end);
      }
    });
  }

  // --- ADDRESS --------------------------------------------------------------

  const STREET_SUFFIX = "Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Marg|Nagar|Colony|Sector|Block|Boulevard|Blvd|Drive|Highway|Hwy|Enclave|Chowk|Bazar|Bazaar|Gali|Galli|Layout|Park|Apartments?|Society|Towers?|Residency|Vihar|Puram|Bagh|Ganj|Sarani";
  const STREET_ADDRESS = new RegExp(
    String.raw`\b\d{1,5}[A-Za-z]?(?:[\/-]\d{1,4}[A-Za-z]?)?,?\s+(?:[A-Z][A-Za-z'’-]*\s+){1,4}(?:` + STREET_SUFFIX +
    String.raw`)\b\.?(?:,\s*(?:[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,2}|\d{6})){0,3}`,
    "g"
  );
  const HOUSE_NUMBER = /\b(?:flat|house\s+no\.?|h\.\s?no\.?|plot\s+no\.?|door\s+no\.?)\s*[:\-#]?\s*\d{1,5}[A-Za-z]?(?:[\/-]\d{1,4}[A-Za-z]?)?/gi;
  const ADDRESS_LABEL = /(?<!(?:e-?mail|ip|mac|web|wallet|url|server|bitcoin|ethereum|hardware|network|return|reply)\s)(?<![A-Za-z])(?:(?:residential|permanent|current|present|mailing|postal|billing|shipping|delivery|home|street|correspondence)\s+)?address\s*[:\-]\s*/gi;
  const POSTAL_CODE = /(?<![A-Za-z])(?:pin\s*code|pincode|postal\s+code|post\s+code|zip\s*code|zip)\s*[:\-]?\s*(\d{6}|\d{5}(?:-\d{4})?)\b/gi;

  function addressRules(text, push) {
    for (const regex of [STREET_ADDRESS, HOUSE_NUMBER]) {
      regex.lastIndex = 0;
      let match;

      while ((match = regex.exec(text)) !== null) {
        push("address", match.index, match.index + match[0].length);
      }
    }

    eachTrigger(ADDRESS_LABEL, text, (from) => {
      if (startsWithPlaceholder(text, from)) {
        return;
      }

      let value = text.slice(from, from + 140);
      const cuts = [
        value.indexOf("\n"),
        value.search(/\s[A-Z][a-z]+\s*:/),
        value.search(/\.\s+[A-Z][a-z]+\s+[a-z]/),
        value.search(/\s\|\s/)
      ].filter((index) => index >= 0);

      if (cuts.length > 0) {
        value = value.slice(0, Math.min(...cuts));
      }

      value = value.replace(/[\s,.;]+$/, "");

      if (value.length >= 6 && /[\d,]/.test(value)) {
        push("address", from, from + value.length);
      }
    });

    POSTAL_CODE.lastIndex = 0;
    let match;

    while ((match = POSTAL_CODE.exec(text)) !== null) {
      const end = match.index + match[0].length;
      push("address", end - match[1].length, end);
    }
  }

  // --- ORGANIZATION ---------------------------------------------------------

  const ORG_CONTEXT = /\b(?:works?\s+(?:at|for|with)|working\s+(?:at|for|with)|worked\s+(?:at|for)|employed\s+(?:at|by|with)|employee\s+of|studies\s+at|studying\s+at|studied\s+at|student\s+(?:at|of)|graduated\s+from|alumn(?:us|a|i)\s+of|interns?\s+at|interning\s+at|hired\s+by|founder\s+of)\s+/gi;
  const ORG_LABEL = /(?<![A-Za-z][ \t]?)(?:company(?:\s+name)?|employer(?:\s+name)?|organi[sz]ation(?:\s+name)?|institution|college|school|university|workplace|business\s+name)\s*[:\-]\s*/gi;
  const ORG_AT = /\b[Aa]t\s+/g;

  function organizationRules(text, push) {
    const accept = (from) => {
      const run = readOrgRun(text, from);

      if (run) {
        push("organization", run.start, run.end);
      }
    };

    eachTrigger(ORG_CONTEXT, text, accept);
    eachTrigger(ORG_LABEL, text, accept);

    // "at Example Technologies" - only when the name ends in a corporate suffix.
    eachTrigger(ORG_AT, text, (from) => {
      const run = readOrgRun(text, from);

      if (run && CORPORATE_SUFFIXES.has(run.tokens[run.tokens.length - 1].toLowerCase())) {
        push("organization", run.start, run.end);
      }
    });
  }

  // --- JOB TITLE ------------------------------------------------------------

  const JOB_LABEL = /(?<![A-Za-z][ \t]?)(?:designation|job\s+title|job\s+role|job\s+profile|occupation|profession|current\s+(?:role|position|designation))\s*[:\-]\s*/gi;
  const JOB_PHRASE = /\b(?:works?\s+as|working\s+as|worked\s+as|employed\s+as|serves\s+as|serving\s+as)\s+(?:an?\s+|the\s+)?/gi;

  function jobTitleRules(text, push) {
    const accept = (from) => {
      const run = readWordRun(text, from, 5);

      if (run) {
        push("job-title", run.start, run.end);
      }
    };

    eachTrigger(JOB_LABEL, text, accept);
    eachTrigger(JOB_PHRASE, text, accept);
  }

  // --- USERNAME -------------------------------------------------------------

  const USERNAME_LABEL = /(?<![A-Za-z][ \t]?)(?:user\s*name|user\s*id|login\s*id|login\s+name|member\s+id|handle|screen\s*name|nick\s*name|display\s+name|gamer\s*tag)\s*[:\-]\s*@?/gi;
  const USERNAME_SESSION = /\b(?:signed\s+in\s+as|logged\s+in\s+as|continue\s+as|continuing\s+as)\s+@?/gi;
  const USERNAME_VALUE = /[A-Za-z0-9](?:[A-Za-z0-9._-]{0,30}[A-Za-z0-9])?/y;
  const SOCIAL_HANDLE = /(?<![A-Za-z0-9._%+\-@/])@[A-Za-z0-9_](?:[A-Za-z0-9_.]{0,28}[A-Za-z0-9_])?(?![A-Za-z0-9_]|\.[A-Za-z])/g;

  function usernameRules(text, push) {
    const accept = (from) => {
      USERNAME_VALUE.lastIndex = from;
      const match = USERNAME_VALUE.exec(text);

      if (!match || match[0].length < 3 || isStopWord(match[0])) {
        return;
      }

      push("username", from, from + match[0].length);
    };

    eachTrigger(USERNAME_LABEL, text, accept);
    eachTrigger(USERNAME_SESSION, text, accept);

    SOCIAL_HANDLE.lastIndex = 0;
    let match;

    while ((match = SOCIAL_HANDLE.exec(text)) !== null) {
      if (match[0].length >= 4) {
        push("username", match.index, match.index + match[0].length);
      }
    }
  }

  // --- MISC PERSONAL ATTRIBUTES ---------------------------------------------

  const MISC_LABEL = /(?<![A-Za-z][ \t]?)(?:gender|sex|blood\s+group|blood\s+type|marital\s+status|nationality|religion|caste|mother\s+tongue|disability|sexual\s+orientation)\s*[:\-]\s*/gi;
  const BLOOD_GROUP = /(?:AB|A|B|O)\s?(?:[+-](?:ve)?|positive|negative)/iy;

  function miscPersonalRules(text, push) {
    eachTrigger(MISC_LABEL, text, (from) => {
      BLOOD_GROUP.lastIndex = from;
      const blood = BLOOD_GROUP.exec(text);

      if (blood) {
        push("misc-personal", from, from + blood[0].length);
        return;
      }

      const run = readWordRun(text, from, 2);

      if (run) {
        push("misc-personal", run.start, run.end);
      }
    });
  }

  // --- DATE / TIME / AGE ----------------------------------------------------

  const MONTH = String.raw`(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?`;
  const DATE_EXPRESSION = new RegExp([
    String.raw`\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?${MONTH},?\s+\d{2,4}\b`,
    String.raw`\b${MONTH}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b`,
    String.raw`\b\d{1,2}[\/.-]\d{1,2}[\/.-](?:\d{4}|\d{2})\b`,
    String.raw`\b\d{4}-\d{2}-\d{2}\b`
  ].join("|"), "gi");
  const TIME_EXPRESSION = /\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\b)?|\b\d{1,2}\s?[AaPp]\.?[Mm]\b/g;

  // Dates and times are only personal when attached to a person-scoped event.
  // A publication date or a clock in the header has no such cue.
  const PERSONAL_EVENT_CONTEXT = /\b(?:appointment|appt\.?|booking|booked\s+(?:for|on)|reservation|reserved\s+(?:for|on)|check[\s-]?in|check[\s-]?out|d\.?o\.?b|date\s+of\s+birth|birth\s*date|birthday|born(?:\s+on)?|anniversary|joined\s+on|joining\s+date|member\s+since|visit(?:ing)?\s+(?:date|on|time)|consultation|scheduled\s+(?:for|on|at)|due\s+date|travel\s+date|journey\s+date|departure|arrival|interview|admission\s+date|admitted\s+on|discharged?\s+on|surgery|delivery\s+(?:date|on|by)|slot|meeting\s+(?:on|at)|pick-?up|drop-?off|test\s+date|exam\s+date|vaccination|dose)\b/gi;
  const AGE_LABEL = /(?<![A-Za-z])(?:age\s*(?:[:\-]|\bis\b)\s*|aged\s+)(\d{1,3})\b/gi;
  const AGE_PHRASE = /\b(\d{1,3})(?:\s*-\s*|\s+)(?:years?|yrs?)(?:\s*-\s*|\s+)old\b|\b(\d{1,3})\s*(?:y\/o|yo)\b/gi;

  function dateTimeAgeRules(text, push) {
    eachTrigger(PERSONAL_EVENT_CONTEXT, text, (from) => {
      let window = text.slice(from, from + 90);
      const cuts = [
        window.indexOf("\n"),
        window.search(/[.!?](?=\s+[A-Z][a-z])/),
        window.indexOf(";"),
        window.search(/\s\|\s/)
      ].filter((index) => index >= 0);

      if (cuts.length > 0) {
        window = window.slice(0, Math.min(...cuts));
      }

      const dateSpans = [];
      DATE_EXPRESSION.lastIndex = 0;
      let match;

      while ((match = DATE_EXPRESSION.exec(window)) !== null) {
        dateSpans.push([match.index, match.index + match[0].length]);
        push("date", from + match.index, from + match.index + match[0].length);
      }

      TIME_EXPRESSION.lastIndex = 0;

      while ((match = TIME_EXPRESSION.exec(window)) !== null) {
        const start = match.index;
        const end = start + match[0].length;

        if (!dateSpans.some(([dateStart, dateEnd]) => start < dateEnd && end > dateStart)) {
          push("time", from + start, from + end);
        }
      }
    });

    for (const regex of [AGE_LABEL, AGE_PHRASE]) {
      regex.lastIndex = 0;
      let match;

      while ((match = regex.exec(text)) !== null) {
        const years = Number(match[1] ?? match[2]);

        if (!(years > 0 && years <= 120)) {
          continue;
        }

        const end = match.index + match[0].length;
        const start = regex === AGE_LABEL ? end - match[1].length : match.index;
        push("age", start, end);
      }
    }
  }

  const ENTITY_RULES = [
    personRules,
    locationRules,
    addressRules,
    organizationRules,
    jobTitleRules,
    usernameRules,
    miscPersonalRules,
    dateTimeAgeRules
  ];

  function collectEntityCandidates(text) {
    const candidates = [];
    const push = (type, start, end) => {
      if (end > start && !startsWithPlaceholder(text, start)) {
        candidates.push({ type, priority: ENTITY_PRIORITY[type], start, end });
      }
    };

    for (const rule of ENTITY_RULES) {
      rule(text, push);
    }

    return candidates;
  }

  function collectPatternCandidates(text) {
    const candidates = [];

    for (const pattern of PII_PATTERNS) {
      pattern.regex.lastIndex = 0;
      let match;

      while ((match = pattern.regex.exec(text)) !== null) {
        const value = match[0];

        if (value.length === 0) {
          pattern.regex.lastIndex += 1;
          continue;
        }

        if (typeof pattern.validate === "function" && !pattern.validate(value)) {
          continue;
        }

        candidates.push({
          type: pattern.type,
          priority: pattern.priority,
          start: match.index,
          end: match.index + value.length
        });
      }
    }

    return candidates;
  }

  // Earliest span wins, then the longest, then the most specific type. A
  // candidate that starts inside an accepted span but runs past its end
  // extends that span, so a partial overlap can never leave a raw tail.
  function resolveCandidates(candidates) {
    candidates.sort((first, second) =>
      first.start - second.start ||
      (second.end - second.start) - (first.end - first.start) ||
      second.priority - first.priority
    );

    const accepted = [];

    for (const candidate of candidates) {
      const last = accepted[accepted.length - 1];

      if (last && candidate.start < last.end) {
        last.end = Math.max(last.end, candidate.end);
        continue;
      }

      accepted.push({ type: candidate.type, start: candidate.start, end: candidate.end });
    }

    return accepted;
  }

  /**
   * Locate PII and contextual personal entities in a string. Returns
   * {type, start, end} only - never the matched text - so callers cannot
   * accidentally forward a value onward.
   *
   * `options.prefix` supplies preceding visible text (e.g. a "Name:" label in a
   * sibling node) as context. Spans are reported relative to `text`, and only
   * spans that reach into `text` are returned.
   */
  function detectPiiMatches(text, options = {}) {
    if (typeof text !== "string" || text.length === 0) {
      return [];
    }

    const prefix = typeof options?.prefix === "string" && options.prefix.length > 0
      ? `${options.prefix.slice(-80)} `
      : "";
    const combined = prefix + text;
    const matches = resolveCandidates([
      ...collectPatternCandidates(combined),
      ...collectEntityCandidates(combined)
    ]);

    if (!prefix) {
      return matches;
    }

    const offset = prefix.length;
    return matches
      .filter((match) => match.end > offset)
      .map((match) => ({
        type: match.type,
        start: Math.max(match.start - offset, 0),
        end: match.end - offset
      }));
  }

  /** Entity-only view, used for demo status counts and focused tests. */
  function detectEntityMatches(text, options = {}) {
    return detectPiiMatches(text, options).filter((match) => match.type in ENTITY_PLACEHOLDERS);
  }

  function containsPii(text) {
    return detectPiiMatches(text).length > 0;
  }

  function placeholderFor(type) {
    return ENTITY_PLACEHOLDERS[type] ?? REDACTION_PLACEHOLDER;
  }

  /** Replace every detected span with a type-appropriate neutral placeholder. */
  function redactPiiText(text) {
    if (typeof text !== "string" || text.length === 0) {
      return text;
    }

    const matches = detectPiiMatches(text);

    if (matches.length === 0) {
      return text;
    }

    let result = "";
    let cursor = 0;

    for (const match of matches) {
      result += text.slice(cursor, match.start) + placeholderFor(match.type);
      cursor = match.end;
    }

    return result + text.slice(cursor);
  }

  /**
   * Deep-scan an outbound payload for surviving PII. Returns findings as
   * {path, type} - never the offending value. `skipKeys` exists so the base64
   * screenshot is not walked: it is already pixel-redacted, and random base64
   * reliably trips the alphanumeric patterns.
   */
  function findPiiInPayload(value, skipKeys = ["image"], path = "") {
    const findings = [];

    if (typeof value === "string") {
      for (const match of detectPiiMatches(value)) {
        findings.push({ path: path || "<root>", type: match.type });
      }
      return findings;
    }

    if (Array.isArray(value)) {
      value.forEach((item, index) => {
        findings.push(...findPiiInPayload(item, skipKeys, `${path}[${index}]`));
      });
      return findings;
    }

    if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        if (skipKeys.includes(key)) {
          continue;
        }
        findings.push(...findPiiInPayload(item, skipKeys, path ? `${path}.${key}` : key));
      }
    }

    return findings;
  }

  /** True when a tag never contributes visible page text. */
  function isNonRenderedTag(tagName) {
    return NON_RENDERED_TAGS.has(String(tagName || "").toLowerCase());
  }

  const api = {
    REDACTION_PLACEHOLDER,
    ENTITY_PLACEHOLDERS,
    NON_RENDERED_TAGS,
    PII_PATTERNS,
    detectPiiMatches,
    detectEntityMatches,
    containsPii,
    redactPiiText,
    placeholderFor,
    findPiiInPayload,
    isNonRenderedTag,
    passesLuhn,
    passesVerhoeff
  };

  globalThis.__privacyVisionPii = api;

  // Available to the Node test runner without turning this into an ES module,
  // which would break its use as a classic content script.
  if (typeof module === "object" && module && module.exports) {
    module.exports = api;
  }
})();
