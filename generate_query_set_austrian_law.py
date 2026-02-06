#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_query_set_austrian_law.py

Synthetic query generator for Austrian-law retrieval/classification.

Design goals
------------
- Exact train/test sizes.
- I.I.D. train/test by default: TRAIN and TEST share the same latent topic mixture (recommended).
- Near-uniform stratification across the same expanded-law label universe.
- High semantic diversity and realistic surface forms.
- Minimal label leakage: law abbreviations are *not* included by default; optional low-probability hints.


Outputs (default filenames)
---------------------------
- train.jsonl
- test.jsonl
- meta.json

Each JSONL row includes:
  query_id, topic_id, query_text, consensus_law, style, issue
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

# -------------------------
# Label universe (84 laws)
# -------------------------
LAWS: List[str] = [
    "ABGB", "APG", "ASVG", "AVG", "AVRAG", "AWG", "AZG", "AktG", "AlVG", "AngG",
    "ArbVG", "AsylG", "AußStrG", "AÜG", "B-VG", "BAO", "BEinstG", "BWG", "DSG", "DSGVO",
    "ECG", "EKHG", "EO", "EStG", "EheG", "ElWOG", "FAGG", "FPG", "FSG", "FinStrG",
    "GSVG", "GewO", "GlBG", "GmbHG", "GrEStG", "IG-L", "IO", "KAKuG", "KBGG", "KFG",
    "KJBG", "KSchG", "KStG", "KartG", "KommStG", "MRG", "MSchG", "MeldeG", "NAG", "NoVAG",
    "PSG", "PatG", "SH-GG", "SMG", "SPG", "SchUG", "StGB", "StPO", "StVO", "StbG",
    "TKG", "TSchG", "UG", "UGB", "UStG", "UVP-G", "UWG", "UrhG", "UrlG", "VKG",
    "VStG", "VbVG", "VerG", "VersG", "VersVG", "VfGG", "VwGG", "VwGVG", "WEG", "WGG",
    "WRG", "WaffG", "ZPO", "ZustG"
]

# -------------------------
# Global enrichment pools
# -------------------------
CITIES_AT = [
    "Wien", "Graz", "Linz", "Salzburg", "Innsbruck", "Klagenfurt", "Wels", "St. Pölten",
    "Dornbirn", "Villach", "Steyr", "Bregenz", "Wiener Neustadt", "Leoben", "Amstetten",
    "Krems", "Ried im Innkreis", "Spittal an der Drau", "Feldkirch", "Schwechat",
]

CHANNELS = [
    "E-Mail", "Brief", "Telefon", "Webformular", "Online-Portal", "persönlich am Schalter",
    "Einschreiben", "Chat", "Hotline",
]

EVIDENCE = [
    "Rechnung", "Vertrag", "Chatverlauf", "E-Mail-Korrespondenz", "Fotos", "Zeug:innen",
    "Kontoauszug", "Bescheid", "Protokoll", "Angebot", "AGB-Auszug", "Lieferschein",
    "Mietvertrag", "Antrag", "Gutachten", "Mahnung",
]

TIME_PHRASES = [
    "heute", "gestern", "letzte Woche", "vor 3 Tagen", "vor 2 Wochen", "vor einem Monat",
    "vor 6 Monaten", "vor 12 Monaten", "vor 18 Monaten",
    "im September 2024", "im November 2025", "Anfang Dezember 2025", "kurz vor Jahresende 2025",
    "im Jänner 2026", "im Februar 2026", "im Frühjahr 2026", "im Sommer 2026",
]


AMOUNTS = [80, 150, 240, 380, 450, 590, 780, 950, 1200, 1750, 2300, 3100, 4800, 5200, 7900, 12500, 19900]

ACTORS = [
    "Konsument:in", "Mieter:in", "Vermieter:in", "Arbeitnehmer:in", "Arbeitgeber:in",
    "Studierende:r", "Unternehmer:in", "Geschäftsführer:in", "Aktionär:in",
    "Lenker:in", "Fahrzeughalter:in", "Beschuldigte:r", "Opfer", "Nachbar:in",
    "Patient:in", "Kund:in", "Handwerker:in",
]

COUNTERPARTIES = [
    "Online-Shop", "Telekom-Anbieter", "Versicherung", "Bank", "Arbeitgeber:in", "Vermieter:in",
    "Behörde", "Gemeinde", "Schule", "Arztpraxis", "Werkstatt", "Reiseveranstalter",
]

AUTHORITIES = [
    "Bezirkshauptmannschaft", "Magistrat", "Finanzamt", "Landesverwaltungsgericht",
    "Bezirksgericht", "Landesgericht", "VwGH", "VfGH",
]

FACETS_COMMON = [
    # Natural facet prompts used for topic expansion (kept as short noun phrases).
    "Voraussetzungen", "Fristen", "Zuständigkeit", "Form/Antrag", "Nachweise/Beweise",
    "Kosten/Gebühren", "Rechtsmittel", "Aufschiebende Wirkung", "Zustellung", "Verjährung",
    "Durchsetzung", "Haftung", "Ausnahmen", "Sanktionen",
]

# Law-lexicon (no law abbreviations). Injected sparingly to increase discriminative signal.
LAW_TERMS: Dict[str, List[str]] = {
    "ABGB": ["Gewährleistung", "Sachmangel", "Rücktritt", "Irrtum", "Schadenersatz", "Verjährung", "Unterhalt", "Besitzstörung"],
    "AktG": ["Hauptversammlung", "Vorstand", "Aufsichtsrat", "Dividende", "Kapitalerhöhung", "Anfechtung", "Satzung", "Aktionärsrechte"],
    "ArbVG": ["Betriebsrat", "Betriebsvereinbarung", "Kollektivvertrag", "Mitbestimmung", "Sozialplan", "Kündigungsanfechtung", "Schlichtung"],
    "ASVG": ["Krankenversicherung", "Pensionsversicherung", "Unfallversicherung", "Krankengeld", "Beitragszeiten", "Selbstversicherung", "ÖGK"],
    "AVG": ["Bescheid", "Parteiengehör", "Akteneinsicht", "Zustellung", "Wiedereinsetzung", "Säumnis", "Ermittlungsverfahren"],
    "BAO": ["Abgabenbescheid", "Einkommensteuer", "Umsatzsteuer", "Säumniszuschlag", "Beschwerde", "Zahlungserleichterung", "Vorauszahlung"],
    "DSG": ["Auskunftsrecht", "Löschung", "Berichtigung", "Einwilligung", "Datenverarbeitung", "Videoüberwachung", "Datenschutzbehörde", "Profiling"],
    "ECG": ["Impressum", "Host-Provider", "Haftung", "Links", "Notice-and-Takedown", "Spam", "Online-Dienst"],
    "EKHG": ["Halterhaftung", "Gefährdungshaftung", "Kfz-Unfall", "Personenschaden", "Mitverschulden", "Haftpflicht"],
    "FAGG": ["Fernabsatz", "Widerruf", "Rücktritt", "Widerrufsfrist", "Informationspflicht", "digitale Inhalte", "Rücksendekosten"],
    "GlBG": ["Diskriminierung", "Belästigung", "Beweislast", "Entschädigung", "Gleichbehandlungskommission", "Benachteiligung"],
    "GmbHG": ["Geschäftsführer", "Generalversammlung", "Stammkapital", "Einlage", "Firmenbuch", "Gesellschafter", "Haftung"],
    "KBGG": ["Kinderbetreuungsgeld", "Karenz", "Zuverdienstgrenze", "Antrag", "Bezugsdauer", "Rückforderung"],
    "KJBG": ["Jugendliche", "Lehrling", "Arbeitszeit", "Nachtarbeit", "Pausen", "Beschäftigungsverbote"],
    "KSchG": ["Verbraucher", "AGB", "Gewährleistung", "Schadenersatz", "Rücktritt", "irreführende Klauseln"],
    "MRG": ["Mietzins", "Befristung", "Kündigung", "Betriebskosten", "Richtwertmiete", "Schlichtungsstelle", "Mietminderung"],
    "MSchG": ["Markenanmeldung", "Widerspruch", "Verwechslungsgefahr", "Nichtigkeit", "Markenverletzung", "Patentamt", "Klassen"],
    "SH-GG": ["Sozialhilfe", "Mindestsicherung", "Anspruch", "Bescheid", "Rückforderung", "Wohnkosten", "Bedarfsprüfung"],
    "StGB": ["Diebstahl", "Betrug", "Körperverletzung", "Nötigung", "Vorsatz", "Notwehr", "Versuch", "Anzeige"],
    "StPO": ["Ermittlungsverfahren", "Beschuldigte", "Verteidigung", "Hausdurchsuchung", "Beschlagnahme", "Untersuchungshaft", "Diversion"],
    "StVO": ["Geschwindigkeitsbegrenzung", "Vorrangregel", "Parkverbot", "Überholverbot", "Abstandsregel", "Rotlicht", "Anhaltepflicht", "Handy am Steuer", "Schutzweg"],
    "TKG": ["Rufnummernmitnahme", "Vertragslaufzeit", "Störung", "Roaming", "Entgelt", "Kündigungsfrist", "Netzbetreiber"],
    "UGB": ["Unternehmer", "Handelsgeschäft", "Rechnung", "Zahlungsverzug", "Prokura", "Firmenbuch", "Rügepflicht"],
    "VKG": ["Väterkarenz", "Papamonat", "Familienzeitbonus", "Karenzmeldung", "Meldefrist", "Vorankündigungsfrist", "Kündigungs- und Entlassungsschutz", "Karenzverlängerung", "Karenzteilung", "Rückkehr an den Arbeitsplatz"],
    "VStG": ["Strafverfügung", "Einspruch", "Organmandat", "Verwaltungsstrafverfahren", "Geldstrafe", "Ersatzfreiheitsstrafe", "Verfolgungsverjährung"],
    "VwGVG": ["Beschwerde", "Landesverwaltungsgericht", "mündliche Verhandlung", "aufschiebende Wirkung", "Erkenntnis", "Revision"],
    "WEG": ["Eigentümergemeinschaft", "Beschluss", "Eigentümerversammlung", "Nutzwert", "Verwaltung", "Anfechtung", "Rücklage"],
    "ZPO": ["Klage", "Zustellung", "Beweisaufnahme", "Versäumungsurteil", "Kosten", "Berufung", "Fristen"],
}

# Optional law-specific context overrides to reduce synthetic artifacts and increase topical realism.

# Additional law-lexicon (expanded law universe). Injected sparingly to increase discriminative signal.
EXTRA_LAW_TERMS: Dict[str, List[str]] = {
    "APG": ["Pensionskonto", "Kontogutschrift", "Korridorpension", "Wartezeit", "Pensionsantritt", "Durchrechnungszeitraum"],
    "AVRAG": ["Dienstzettel", "Betriebsübergang", "Entsendung", "Informationspflicht", "Kündigungsschutz", "Arbeitsvertrag"],
    "AWG": ["Abfallbesitzer", "Entsorgungspflicht", "gefährlicher Abfall", "Altstoffsammelzentrum", "Abfallbilanz", "Abfallbeauftragte:r"],
    "AZG": ["Arbeitszeit", "Überstunden", "Ruhezeit", "Gleitzeit", "Arbeitszeitaufzeichnung", "Höchstarbeitszeit"],
    "AlVG": ["Arbeitslosengeld", "Notstandshilfe", "Sperrfrist", "AMS", "Zumutbarkeit", "Meldepflicht"],
    "AngG": ["Kündigungsfrist", "Entlassung", "Provisionsanspruch", "Konkurrenzklausel", "Abfertigung", "Dienstverhinderung"],
    "AsylG": ["Asylantrag", "Status", "Subsidiärer Schutz", "Dublin", "BFA", "Rückkehrentscheidung"],
    "AußStrG": ["Pflegschaft", "Erwachsenenvertretung", "Verlassenschaft", "Beschluss", "Rekurs", "Abänderungsantrag"],
    "AÜG": ["Arbeitskräfteüberlassung", "Überlasser", "Beschäftiger", "Equal Pay", "Überlassungsvertrag", "Sanktionen"],
    "B-VG": ["Grundrechte", "Gleichheitsgrundsatz", "Kompetenzverteilung", "Verhältnismäßigkeit", "Gesetzesprüfung", "Verordnung"],
    "BEinstG": ["Begünstigte Behinderte", "Ausgleichstaxe", "Kündigungsschutz", "Einstellungspflicht", "Behindertenpass", "Behindertenausschuss"],
    "BWG": ["Bankgeheimnis", "KYC", "Kreditvertrag", "Kontosperre", "Beschwerde", "Eigenmittel"],
    "DSGVO": ["Auskunft", "Löschung", "Widerspruch", "Auftragsverarbeitung", "Datenpanne", "Drittlandtransfer", "DSFA"],
    "EO": ["Exekutionstitel", "Lohnpfändung", "Fahrnisexekution", "Zwangsversteigerung", "Exekutionsbewilligung", "Einstellung"],
    "EStG": ["Werbungskosten", "Sonderausgaben", "außergewöhnliche Belastung", "Pendlerpauschale", "Familienbonus", "Arbeitnehmerveranlagung", "AfA"],
    "EheG": ["einvernehmliche Scheidung", "Verschuldensscheidung", "Unterhalt", "Aufteilung", "Ehewohnung", "Obsorge"],
    "ElWOG": ["Netzzugang", "Grundversorgung", "Netzentgelte", "Lieferantenwechsel", "E-Control", "Stromabschaltung"],
    "FPG": ["Einreiseverbot", "Rückkehrentscheidung", "Schubhaft", "Abschiebung", "Duldung", "Ausweisung"],
    "FSG": ["Lenkberechtigung", "Entzug", "Vormerksystem", "Nachschulung", "Probezeit", "medizinisches Gutachten"],
    "FinStrG": ["Abgabenhinterziehung", "Selbstanzeige", "Finanzstrafverfahren", "Geldstrafe", "Zoll", "Vorsatz"],
    "GSVG": ["SVS", "Pflichtversicherung", "Beitragsgrundlage", "Mindestbeitrag", "Nachbemessung", "Krankenversicherung"],
    "GewO": ["Gewerbeanmeldung", "Befähigungsnachweis", "Betriebsanlage", "Gastgewerbe", "Gewerbeberechtigung", "Gewerberechtlicher Geschäftsführer"],
    "GrEStG": ["Grunderwerbsteuer", "Bemessungsgrundlage", "Befreiung", "Selbstberechnung", "Schenkung", "Immobilienkauf"],
    "IG-L": ["Feinstaub", "Fahrverbot", "Luftreinhalteplan", "Ausnahme", "Kontrolle", "Strafe"],
    "IO": ["Insolvenz", "Sanierungsverfahren", "Schuldenregulierung", "Zahlungsplan", "Abschöpfungsverfahren", "Forderungsanmeldung"],
    "KAKuG": ["Krankenanstalt", "Aufnahme", "Pflegegebühren", "Kostenersatz", "Entlassung", "Patient:innenrechte"],
    "KFG": ["Zulassung", "Pickerl", "Halterpflicht", "Umbau", "Lenkerauskunft", "technischer Mangel"],
    "KStG": ["Körperschaftsteuer", "Mindest-KöSt", "Gruppenbesteuerung", "verdeckte Ausschüttung", "Gewinnausschüttung", "Abzugsfähigkeit"],
    "KartG": ["Kartellverbot", "Fusionskontrolle", "Marktbeherrschung", "BWB", "Kartellgericht", "Bußgeld"],
    "KommStG": ["Kommunalsteuer", "Lohnsumme", "Betriebsstätte", "Haftung", "Abfuhr", "Prüfung"],
    "MeldeG": ["Hauptwohnsitz", "Meldezettel", "Anmeldung", "Abmeldung", "Frist", "Meldepflicht"],
    "NAG": ["Aufenthaltstitel", "Rot-Weiß-Rot Karte", "Familiennachzug", "Verlängerung", "Daueraufenthalt EU", "Erwerbstätigkeit"],
    "NoVAG": ["Normverbrauchsabgabe", "Kfz-Import", "Bemessung", "Befreiung", "Nachzahlung", "E-Auto"],
    "PSG": ["Privatstiftung", "Stiftungsurkunde", "Begünstigte", "Stiftungsvorstand", "Widerruf", "Stiftungszweck"],
    "PatG": ["Patent", "Neuheit", "Erfinder", "Lizenz", "Patentverletzung", "Nichtigkeit"],
    "SMG": ["Suchtmittel", "Besitz", "Eigengebrauch", "Diversion", "Therapie", "Sicherstellung"],
    "SPG": ["Wegweisung", "Betretungsverbot", "Identitätsfeststellung", "Anhaltung", "Gefährderansprache", "Waffenverbot"],
    "SchUG": ["Beurteilung", "Schularbeit", "Wiederholungsprüfung", "Ausschluss", "Disziplinarmaßnahme", "Schulpflicht"],
    "StbG": ["Einbürgerung", "Verleihung", "Entziehung", "Doppelstaatsbürgerschaft", "Deutschkenntnisse", "Strafregister"],
    "TSchG": ["Tierhaltung", "Tierquälerei", "Hundehaltung", "Tierversuch", "Haltungsbedingungen", "Strafe"],
    "UG": ["Zulassung", "Studienbeitrag", "Prüfungswiederholung", "Anerkennung", "Beurlaubung", "Disziplinarverfahren"],
    "UStG": ["Vorsteuerabzug", "UID", "Reverse Charge", "Kleinunternehmer", "UVA", "innergemeinschaftliche Lieferung", "Rechnung"],
    "UVP-G": ["Umweltverträglichkeitsprüfung", "UVP-Pflicht", "Bürgerbeteiligung", "Genehmigungsverfahren", "Umweltanwalt", "Beschwerde"],
    "UWG": ["irreführende Werbung", "vergleichende Werbung", "Unterlassung", "aggressive Geschäftspraktiken", "Schadenersatz", "Abmahnung"],
    "UrhG": ["Urheberrecht", "Nutzungsrecht", "Zitatrecht", "Plagiat", "Lichtbild", "Lizenz"],
    "UrlG": ["Urlaubsanspruch", "Urlaubsverbrauch", "Urlaubsersatzleistung", "Urlaubssperre", "Verjährung", "Urlaubsvereinbarung"],
    "VbVG": ["Verbandsverantwortlichkeit", "Compliance", "Verbandsgeldbuße", "Leitungsperson", "Organisationsverschulden", "Diversion"],
    "VerG": ["Vereinsgründung", "Statuten", "Vereinsregister", "Vorstand", "Auflösung", "Vereinsbehörde"],
    "VersG": ["Versammlung anmelden", "Spontanversammlung", "Untersagung", "Auflagen", "Ordner", "Störung"],
    "VersVG": ["Versicherungsvertrag", "Rücktritt", "Anzeigepflicht", "Obliegenheit", "Prämienverzug", "Schadensmeldung"],
    "VfGG": ["Beschwerde VfGH", "Individualantrag", "Gesetzesprüfung", "Verfahrenshilfe", "Frist", "Zuständigkeit"],
    "VwGG": ["Revision VwGH", "Zulässigkeit", "Frist", "aufschiebende Wirkung", "Begründung", "Verfahrenshilfe"],
    "WGG": ["gemeinnützige Bauvereinigung", "Mietzins", "Betriebskosten", "Finanzierungsbeitrag", "Kaution", "Wohnungszuweisung"],
    "WRG": ["Wasserrecht", "Bewilligung", "Brunnen", "Einleitung", "Uferstreifen", "Hochwasserschutz"],
    "WaffG": ["Waffenbesitzkarte", "Waffenpass", "Verbotene Waffen", "Aufbewahrung", "Waffenverbot", "Zuverlässigkeit"],
    "ZustG": ["Zustellung", "Hinterlegung", "RSb/RSa", "elektronische Zustellung", "Zustellmangel", "Fristbeginn"],
}

def _merge_terms(dst: Dict[str, List[str]], src: Dict[str, List[str]]) -> None:
    for law, terms in src.items():
        if law not in dst:
            dst[law] = list(terms)
        else:
            for t in terms:
                if t not in dst[law]:
                    dst[law].append(t)

_merge_terms(LAW_TERMS, EXTRA_LAW_TERMS)

LAW_CONTEXT_OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "DSG": {"authorities": ["Datenschutzbehörde", "DSB", "Magistrat (Datenschutz)"], "counterparties": ["Arbeitgeber:in", "Online-Plattform", "Behörde"]},
    "BAO": {"authorities": ["Finanzamt", "Bundesfinanzgericht"], "counterparties": ["Finanzamt", "Arbeitgeber:in", "Bank"]},
    "ASVG": {"authorities": ["ÖGK", "PVA", "SVS"], "counterparties": ["ÖGK", "Arbeitgeber:in"]},
    "MRG": {"authorities": ["Schlichtungsstelle", "Bezirksgericht"], "counterparties": ["Vermieter:in", "Hausverwaltung"]},
    "WEG": {"authorities": ["Hausverwaltung", "Bezirksgericht"], "counterparties": ["Eigentümergemeinschaft", "Hausverwaltung"]},
    "VStG": {"authorities": ["Magistrat", "Bezirkshauptmannschaft"], "counterparties": ["Behörde", "Magistrat"]},
    "VwGVG": {"authorities": ["Landesverwaltungsgericht", "VwGH"], "counterparties": ["Behörde", "Bezirkshauptmannschaft"]},
    "StPO": {"authorities": ["Polizei", "Staatsanwaltschaft", "Landesgericht"], "counterparties": ["Polizei", "Staatsanwaltschaft"]},
    "StGB": {"authorities": ["Polizei", "Staatsanwaltschaft", "Landesgericht"], "counterparties": ["Polizei"]},
    "StVO": {"authorities": ["Magistrat", "Bezirkshauptmannschaft"], "counterparties": ["Polizei"]},
    "TKG": {"authorities": ["RTR", "Schlichtungsstelle"], "counterparties": ["Telekom-Anbieter"]},
    "VKG": {"actors": ["Vater (Arbeitnehmer)", "zweiter Elternteil", "Arbeitnehmer:in"], "counterparties": ["Arbeitgeber:in"], "authorities": ["Arbeits- und Sozialgericht", "Arbeits- und Sozialgericht Wien"]},
}


# Additional law-specific context overrides (expanded law universe).
EXTRA_LAW_CONTEXT_OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "AWG": {"counterparties": ["Entsorgungsbetrieb", "Behörde"], "authorities": ["Bezirkshauptmannschaft", "Umweltbehörde"]},
    "AsylG": {"counterparties": ["BFA", "Behörde"], "authorities": ["BFA", "Bundesverwaltungsgericht"]},
    "AußStrG": {"counterparties": ["Verfahrenspartei"], "authorities": ["Bezirksgericht"]},
    "BAO": {"counterparties": ["Finanzamt", "Arbeitgeber:in", "Bank"], "authorities": ["Finanzamt", "Bundesfinanzgericht"]},
    "BWG": {"counterparties": ["Bank", "Kund:in"], "authorities": ["FMA", "OeNB"]},
    "DSGVO": {"counterparties": ["Arbeitgeber:in", "Online-Plattform", "Behörde"], "authorities": ["Datenschutzbehörde", "DSB"]},
    "EO": {"counterparties": ["Gläubiger:in", "Schuldner:in"], "authorities": ["Bezirksgericht", "Exekutionsgericht"]},
    "EStG": {"counterparties": ["Finanzamt", "Arbeitgeber:in", "Steuerberater:in"], "authorities": ["Finanzamt", "Bundesfinanzgericht"]},
    "ElWOG": {"counterparties": ["Netzbetreiber", "Stromanbieter"], "authorities": ["E-Control", "Schlichtungsstelle"]},
    "FPG": {"counterparties": ["BFA", "Polizei"], "authorities": ["BFA", "Landesverwaltungsgericht", "Polizei"]},
    "FSG": {"counterparties": ["Behörde", "Polizei"], "authorities": ["Bezirkshauptmannschaft", "Magistrat"]},
    "FinStrG": {"counterparties": ["Finanzamt", "Zoll", "Behörde"], "authorities": ["Finanzamt", "Finanzstrafbehörde", "Bundesfinanzgericht"]},
    "GrEStG": {"counterparties": ["Notar:in", "Finanzamt", "Verkäufer:in"], "authorities": ["Finanzamt", "Bundesfinanzgericht"]},
    "IO": {"counterparties": ["Insolvenzverwalter:in", "Gläubiger:in"], "authorities": ["Insolvenzgericht", "Landesgericht"]},
    "KAKuG": {"counterparties": ["Krankenanstalt", "Patient:in"], "authorities": ["Krankenanstalt", "Patientenanwaltschaft"]},
    "KFG": {"counterparties": ["Polizei", "Zulassungsstelle"], "authorities": ["Bezirkshauptmannschaft", "Magistrat", "Polizei"]},
    "KStG": {"counterparties": ["Finanzamt", "GmbH", "AG"], "authorities": ["Finanzamt", "Bundesfinanzgericht"]},
    "KartG": {"counterparties": ["Unternehmen", "BWB"], "authorities": ["Bundeswettbewerbsbehörde", "Kartellgericht"]},
    "KommStG": {"counterparties": ["Gemeinde", "Arbeitgeber:in"], "authorities": ["Gemeinde", "Stadtmagistrat"]},
    "NAG": {"counterparties": ["Behörde", "Arbeitgeber:in"], "authorities": ["Bezirkshauptmannschaft", "MA 35", "BFA"]},
    "NoVAG": {"counterparties": ["Autohändler:in", "Finanzamt"], "authorities": ["Finanzamt"]},
    "PatG": {"counterparties": ["Patentamt", "Lizenznehmer:in"], "authorities": ["Patentamt", "Oberlandesgericht"]},
    "SPG": {"counterparties": ["Polizei"], "authorities": ["Polizei", "Bezirkshauptmannschaft"]},
    "SchUG": {"counterparties": ["Schule"], "authorities": ["Schulleitung", "Bildungsdirektion"]},
    "StbG": {"counterparties": ["Behörde"], "authorities": ["Bezirkshauptmannschaft", "Magistrat", "Landesregierung"]},
    "TSchG": {"counterparties": ["Tierhalter:in"], "authorities": ["Bezirkshauptmannschaft", "Amtstierarzt"]},
    "UG": {"counterparties": ["Universität"], "authorities": ["Universität", "Studienpräses"]},
    "UStG": {"counterparties": ["Finanzamt", "Kund:in", "Unternehmer:in"], "authorities": ["Finanzamt", "Bundesfinanzgericht"]},
    "UVP-G": {"counterparties": ["Projektwerber:in", "Behörde"], "authorities": ["UVP-Behörde", "Landesregierung"]},
    "UWG": {"counterparties": ["Mitbewerber:in", "Unternehmen"], "authorities": ["Handelsgericht", "Bezirksgericht"]},
    "UrhG": {"counterparties": ["Plattform", "Nutzer:in"], "authorities": ["Bezirksgericht", "Handelsgericht"]},
    "VerG": {"counterparties": ["Vereinsbehörde"], "authorities": ["Vereinsbehörde", "Landespolizeidirektion"]},
    "VersG": {"counterparties": ["Polizei"], "authorities": ["Polizei", "Bezirkshauptmannschaft", "Magistrat"]},
    "WRG": {"counterparties": ["Behörde"], "authorities": ["Wasserrechtsbehörde", "Bezirkshauptmannschaft"]},
    "ZustG": {"counterparties": ["Behörde"], "authorities": ["Behörde", "Post"]},
}

def _merge_overrides(dst: Dict[str, Dict[str, List[str]]], src: Dict[str, Dict[str, List[str]]]) -> None:
    for law, ov in src.items():
        if law not in dst:
            dst[law] = {k: list(v) for k, v in ov.items()}
            continue
        for k, v in ov.items():
            if k not in dst[law]:
                dst[law][k] = list(v)
            else:
                for item in v:
                    if item not in dst[law][k]:
                        dst[law][k].append(item)

_merge_overrides(LAW_CONTEXT_OVERRIDES, EXTRA_LAW_CONTEXT_OVERRIDES)



# -------------------------
# Text utilities
# -------------------------
_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+", re.UNICODE)
_PLACEHOLDER_RE = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)}")

STOPWORDS = {
    "welche", "welcher", "welches", "wann", "wie", "wo", "was", "ist", "sind",
    "ich", "mein", "meine", "mich", "mir", "wir", "uns", "ihr", "sie", "man",
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer",
    "und", "oder", "für", "bei", "von", "in", "im", "auf", "nach", "gegen", "ohne",
    "österreich", "recht", "gesetz", "paragraph", "paragraf", "at",
}

def stable_int(s: str) -> int:
    acc = 0
    for i, ch in enumerate(s, start=1):
        acc = (acc * 131 + ord(ch) * i) & 0xFFFFFFFF
    return acc

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_keywords(source: str, max_tokens: int = 10) -> str:
    toks: List[str] = []
    for t in _WORD_RE.findall(source.replace("-", " ").replace("-", " ")):
        tl = t.lower().strip(".,;:!?()[]{}\"'“”„")
        if not tl or tl in STOPWORDS:
            continue
        if tl.isdigit() or len(tl) > 2:
            toks.append(tl)
    return " ".join(toks[:max_tokens]) if toks else source[:50].lower()

def maybe_apply_surface_noise(text: str, rng: random.Random, p: float) -> str:
    """Moderate surface noise to improve robustness while keeping TF-IDF/SVD usable.

    Note: keep p small; noise is applied *after* optional hint injection.
    """
    if rng.random() >= p:
        return text

    def typo_once(s: str) -> str:
        # Single lightweight typo in a non-trivial token.
        toks = s.split()
        cand_idx = [i for i,t in enumerate(toks) if len(t) >= 6 and t.isalpha()]
        if not cand_idx:
            return s
        i = rng.choice(cand_idx)
        w = toks[i]
        j = rng.randrange(1, len(w)-1)
        op = rng.choice(["del", "dup", "swap"])
        if op == "del":
            w2 = w[:j] + w[j+1:]
        elif op == "dup":
            w2 = w[:j] + w[j] + w[j:]
        else:
            w2 = w[:j-1] + w[j] + w[j-1] + w[j+1:]
        toks[i] = w2
        return " ".join(toks)

    variants = [
        text.replace("Österreich", "AT"),
        text.replace("E-Mail", "Email"),
        text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"),
        text.rstrip("?") if text.endswith("?") else text + "?",
        text.lower() if rng.random() < 0.7 else text,
        typo_once(text) if rng.random() < 0.5 else text,
    ]
    v = rng.choice(variants)
    return normalize_ws(v)

def inject_law_hint(text: str, law: str, rng: random.Random) -> str:
    forms = [
        f" (nach {law})",
        f" - {law}",
        f" (gemäß {law})",
        f" unter Bezug auf {law}",
    ]
    return text + rng.choice(forms)

# -------------------------
# Law-specific specs
# -------------------------

@dataclass
class LawSpec:
    templates: List[str]
    slots: Dict[str, List[str]]

def base_law_specs() -> Dict[str, LawSpec]:
    """Base frames (compact), later expanded automatically with facet-combinatorics."""
    goods = [
        "Smartphone", "Laptop", "Fahrrad", "E-Bike", "Kühlschrank", "Waschmaschine",
        "Gebrauchtwagen", "Möbelstück", "Heizgerät", "Software-Abo", "Streaming-Abo",
        "Küchengerät", "Kamera", "Winterreifen",
    ]
    services = [
        "Handwerkerleistung", "Reparatur", "Online-Kurs", "Fitnessvertrag", "Mobilfunkvertrag",
        "Internetvertrag", "Umzug", "Reisebuchung", "Werkstattauftrag",
    ]
    defects = [
        "verstecktem Mangel", "Defekt nach kurzer Zeit", "fehlender zugesicherter Eigenschaft",
        "Lieferverzug", "Falschlieferung", "unvollständiger Lieferung",
    ]
    payments = ["Anzahlung", "Teilzahlung", "Ratenzahlung", "Kaution", "Überweisung", "Barzahlung"]
    employment = ["Kündigung", "Entlassung", "Versetzung", "Überstunden", "Arbeitszeit", "Urlaub", "Abmahnung"]
    discrimination = ["Geschlecht", "Alter", "ethnischer Herkunft", "Religion", "Behinderung", "sexueller Orientierung"]
    admin_subject = [
        "Gewerbeberechtigung", "Meldezettel", "Parkausweis", "Baubescheid", "Verwaltungsstrafe",
        "Führerscheinmaßnahme", "Gastgewerbeberechtigung", "Baugenehmigung",
    ]
    traffic = [
        "Geschwindigkeitsüberschreitung", "Rotlichtverstoß", "Handy am Steuer", "Falschparken",
        "Vorrangverletzung", "Alkohol am Steuer", "Unfall mit Sachschaden", "Unfall mit Personenschaden",
    ]

    specs: Dict[str, LawSpec] = {}

    specs["ABGB"] = LawSpec(
        templates=[
            "Gewährleistung bei {good} wegen {defect}",
            "Rücktritt vom Kaufvertrag über {good} wegen {defect}",
            "Schadenersatz wegen Beschädigung von {good}",
            "Anfechtung eines Vertrags wegen Irrtum im Zusammenhang mit {good}",
            "Rückforderung einer irrtümlichen {payment}",
            "Verjährung bei Schadenersatzanspruch nach {event}",
            "Unterhaltspflichten zwischen {family}",
            "Besitzstörung: {neighbor_conflict}",
        ],
        slots={
            "good": goods,
            "defect": defects,
            "payment": payments,
            "event": ["Unfall", "Sachbeschädigung", "Vertragsverletzung", "Lieferverzug"],
            "family": ["Eltern und Kindern", "geschiedenen Ehepartnern", "Kindern und Großeltern"],
            "neighbor_conflict": ["Zaun versetzt", "Parkplatz blockiert", "Lärm und Immissionen", "Baum ragt über Grundstück"],
        },
    )

    specs["KSchG"] = LawSpec(
        templates=[
            "Unfaire AGB-Klausel in {service} (Konsument:in vs Unternehmen)",
            "Gewährleistung/Schadenersatz bei mangelhafter {service}",
            "Irreführende Werbung und Vertragsabschluss bei {service}",
            "Inkasso- und Mahnspesen bei {service}",
            "Preiserhöhungsklausel bei {service}",
            "Kündigungsfrist und Vertragsbindung bei {service}",
            "Haustürgeschäft: {doorstep}",
        ],
        slots={
            "service": services,
            "doorstep": ["überrumpelnder Abschluss", "fehlende Belehrung", "Widerruf und Rückabwicklung"],
        },
    )

    specs["FAGG"] = LawSpec(
        templates=[
            "Widerrufsrecht bei Online-Kauf von {good}",
            "Widerrufsfrist und Belehrung bei {service} online abgeschlossen",
            "Ausnahmen vom Widerruf bei {exception}",
            "Rücksendekosten und Rückerstattung bei Online-Bestellung von {good}",
            "Digitale Inhalte: Widerruf bei {digital}",
        ],
        slots={
            "good": goods,
            "service": services,
            "exception": ["maßgefertigter Ware", "schnell verderblicher Ware", "versiegelter Ware", "Ticketkauf"],
            "digital": ["Streaming-Abo", "Software-Download", "Online-Kurs", "In-App-Kauf"],
        },
    )

    specs["MRG"] = LawSpec(
        templates=[
            "Mieterhöhung/Indexanpassung im Mietverhältnis",
            "Betriebskostenabrechnung: {bk_issue}",
            "Mängel in der Wohnung: {mangel}",
            "Kaution und Rückzahlung nach Auszug",
            "Kündigung durch Vermieter: {kuendigungsgrund}",
            "Untervermietung: {untervermietung}",
            "Erhaltungsarbeiten: {erhaltung}",
        ],
        slots={
            "bk_issue": ["Nachzahlung", "Einsicht in Belege", "unplausible Positionen", "Aufteilungsschlüssel"],
            "mangel": ["Schimmel", "Heizungsdefekt", "Wasserschaden", "Lärmbelastung", "undichte Fenster"],
            "kuendigungsgrund": ["Eigenbedarf", "Zahlungsverzug", "vertragswidriger Gebrauch", "Scheinmiete"],
            "untervermietung": ["Zustimmung erforderlich", "Untermietzins", "Kurzzeitvermietung"],
            "erhaltung": ["Lift", "Heizung", "Fenster", "Leitungen", "Fassade"],
        },
    )

    specs["WEG"] = LawSpec(
        templates=[
            "Beschluss der Eigentümergemeinschaft zu {measure}",
            "Anfechtung eines WEG-Beschlusses wegen {reason}",
            "Kostenverteilung/Schlüssel bei {measure}",
            "Hausordnung: {house_rule} und Durchsetzung",
            "Nutzwert und Abrechnung: {nutzwert}",
        ],
        slots={
            "measure": ["Dachsanierung", "Fassadensanierung", "Lift-Einbau", "Fenstertausch", "Heizungsumstellung", "Photovoltaik am Dach"],
            "reason": ["Formfehler", "grober Benachteiligung", "fehlender Beschlussfähigkeit", "unzulässiger Mehrheit"],
            "house_rule": ["Lärmzeiten", "Nutzung von Gemeinschaftsflächen", "Haustierhaltung", "Kurzzeitvermietung"],
            "nutzwert": ["Änderung", "Feststellung", "Streit über Schlüssel", "Umbau-Auswirkungen"],
        },
    )

    specs["UGB"] = LawSpec(
        templates=[
            "Zahlungsverzug im B2B-Geschäft: {b2b}",
            "Gewährleistung und Mängelrüge im Unternehmensgeschäft bei {good}",
            "Vertretungsbefugnis/Prokura und {ugb_detail}",
            "Firmenbuch: Eintragung/Änderung und {ugb_detail}",
            "Handelskauf: Lieferverzug und Vertragsstrafe bei {good}",
        ],
        slots={
            "b2b": ["Verzugszinsen", "Mahnspesen", "Eigentumsvorbehalt", "Lieferstopp"],
            "good": goods,
            "ugb_detail": ["Zuständigkeit", "Fristen", "Formvorschriften", "Haftung", "Beweis"],
        },
    )

    specs["GmbHG"] = LawSpec(
        templates=[
            "Bestellung/Abberufung von Geschäftsführer:in und {gmbh_issue}",
            "Haftung von Geschäftsführer:in bei {gmbh_issue}",
            "Gesellschafterbeschluss: {gmbh_vote}",
            "Einlagenleistung und Nachschusspflicht",
            "Firmenbuchänderung: {gmbh_change}",
        ],
        slots={
            "gmbh_issue": ["Insolvenzreife", "Sorgfaltspflichten", "Verstoß gegen Weisungen", "Verbotene Einlagenrückgewähr"],
            "gmbh_vote": ["Stimmverbot", "Einberufung", "Formmängel", "Anfechtung"],
            "gmbh_change": ["Sitzverlegung", "Geschäftsanschrift", "Kapitalerhöhung", "Firmenwortlaut"],
        },
    )

    specs["AktG"] = LawSpec(
        templates=[
            "Hauptversammlung: {hv_issue}",
            "Anfechtung eines HV-Beschlusses wegen {hv_reason}",
            "Aufsichtsrat: {ar_issue}",
            "Aktienübertragung und {akt_issue}",
            "Kapitalmaßnahmen: {kapital}",
        ],
        slots={
            "hv_issue": ["Einberufungsfrist", "Tagesordnung", "Stimmrechtsvertretung", "Auskunftsrecht"],
            "hv_reason": ["Formfehler", "Verfahrensfehler", "unzutreffender Information", "Gesetzesverstoß"],
            "ar_issue": ["Bestellung", "Haftung", "Vergütung", "Interessenkonflikt"],
            "akt_issue": ["Vinkulierung", "Depotübertrag", "Namensaktien", "Sperrfrist"],
            "kapital": ["Kapitalerhöhung", "Kapitalherabsetzung", "Bezugsrecht", "Squeeze-out"],
        },
    )

    specs["ArbVG"] = LawSpec(
        templates=[
            "Betriebsrat: {br_issue}",
            "Betriebsvereinbarung zu {br_topic}",
            "Mitbestimmung bei {employment}",
            "Anfechtung einer Kündigung wegen {br_reason}",
            "Kontrollmaßnahmen/Datenschutz im Betrieb und {br_topic}",
        ],
        slots={
            "br_issue": ["Wahl", "Freistellung", "Informationsrechte", "Betriebsversammlung"],
            "br_topic": ["Arbeitszeit", "Homeoffice", "Überstunden", "Prämien", "Kontrollmaßnahmen"],
            "employment": employment,
            "br_reason": ["Sozialwidrigkeit", "Motivwidrigkeit", "Formmangel", "Betriebsratsanhörung"],
        },
    )

    specs["ASVG"] = LawSpec(
        templates=[
            "Sozialversicherung: Anmeldung bei {employment_form}",
            "Krankengeld bei {health_case}",
            "Arbeitsunfall: {work_accident}",
            "Pension: {pension_case}",
            "Mitversicherung von Angehörigen und {sv_detail}",
        ],
        slots={
            "employment_form": ["Vollzeit", "Teilzeit", "geringfügiger Beschäftigung", "Praktikum", "befristetem Dienstvertrag"],
            "health_case": ["Arbeitsunfähigkeit", "langem Krankenstand", "Reha", "Arbeitsunfall"],
            "work_accident": ["Meldepflicht", "Leistungen", "Abgrenzung zu Freizeitunfall", "Reha-Maßnahmen"],
            "pension_case": ["Alterspension", "Invalidität", "Berufsunfähigkeit", "Ausgleichszulage"],
            "sv_detail": ["Beitragsnachzahlung", "Versicherungszeiten", "Selbstversicherung", "Nachsicht"],
        },
    )

    specs["MSchG"] = LawSpec(
        templates=[
            "Mutterschutz: Beschäftigungsbeschränkung bei {work}",
            "Kündigungs-/Entlassungsschutz in der Schwangerschaft",
            "Mutterschutzfrist vor/nach Geburt und {mschg_detail}",
            "Meldung der Schwangerschaft an Arbeitgeber:in und {mschg_detail}",
            "Karenz und Rückkehrrecht: {karenz}",
        ],
        slots={
            "work": ["Nachtarbeit", "schwerer körperlicher Arbeit", "Gefahrstoffen", "Steharbeit", "Schichtbetrieb"],
            "mschg_detail": ["ärztliches Attest", "Arbeitsplatzanpassung", "Freistellung", "Entgeltfortzahlung"],
            "karenz": ["Teilzeit", "Kündigungsschutz", "Vereinbarung", "Fristen"],
        },
    )

    specs["KBGG"] = LawSpec(
        templates=[
            "Kinderbetreuungsgeld: Anspruch bei {family}",
            "Fristen für Antrag und Nachweise beim Kinderbetreuungsgeld",
            "Partnerschaftsbonus und {kbgg_detail}",
            "Zuverdienstgrenze beim Kinderbetreuungsgeld",
            "Rückforderung/Überprüfung von Kinderbetreuungsgeld wegen {kbgg_reason}",
            "Wechsel des Bezugs zwischen Elternteilen und {kbgg_detail}",
        ],
        slots={
            "family": ["Alleinerziehenden", "Ehepaar", "Lebensgemeinschaft", "getrennt lebenden Eltern"],
            "kbgg_detail": ["Wechsel", "Kontrolle", "Nachweise", "Auszahlung", "Antragsweg"],
            "kbgg_reason": ["Zuverdienst", "Fristversäumnis", "fehlendem Nachweis", "falschen Angaben"],
        },
    )

    specs["KJBG"] = LawSpec(
        templates=[
            "Beschäftigung von Jugendlichen: Arbeitszeit bei {age_group}",
            "Nachtarbeit/Überstundenverbot bei {age_group} in {sector}",
            "Praktikum/Ferialjob: {kjbg_issue}",
            "Verbot gefährlicher Arbeit: {hazard} für {age_group}",
        ],
        slots={
            "age_group": ["15-16 Jahren", "16-17 Jahren", "unter 15 Jahren", "Lehrlingen"],
            "sector": ["Gastronomie", "Einzelhandel", "Bau", "Pflege", "Eventbranche"],
            "kjbg_issue": ["Pausenregelung", "Schulbesuch", "Arbeitsmedizin", "Bewilligungspflicht"],
            "hazard": ["Maschinenarbeit", "Chemikalien", "schweren Lasten", "Arbeiten in großer Höhe"],
        },
    )

    specs["GlBG"] = LawSpec(
        templates=[
            "Diskriminierung im Job wegen {discrimination}",
            "Benachteiligung bei Bewerbung/Entlohnung wegen {discrimination}",
            "Belästigung am Arbeitsplatz und {glbg_detail}",
            "Beweislast/Schadenersatz bei Diskriminierung wegen {discrimination}",
        ],
        slots={
            "discrimination": discrimination,
            "glbg_detail": ["Dokumentation", "Fristen", "Schlichtung", "Geltendmachung", "Schadenersatz"],
        },
    )

    specs["DSG"] = LawSpec(
        templates=[
            "Auskunftsrecht nach DSGVO bei {controller}",
            "Löschung/Berichtigung von Daten bei {controller}",
            "Videoüberwachung und Datenschutz in {place}",
            "Datenpanne/Meldepflicht bei {controller}",
            "Rechtsgrundlage für Verarbeitung (Einwilligung/Vertrag) bei {controller}",
        ],
        slots={
            "controller": ["Online-Shop", "Arbeitgeber:in", "Schule", "Arztpraxis", "Gemeinde", "Telekom-Anbieter", "Fitnessstudio", "Versicherung"],
            "place": ["Mehrparteienhaus", "Geschäftslokal", "Arbeitsplatz", "Schule", "Parkplatz"],
        },
    )

    specs["TKG"] = LawSpec(
        templates=[
            "Kündigungsfrist/Vertragsbindung bei {tel_service}",
            "Rufnummernmitnahme und {tkg_issue}",
            "Störung/Leistungsausfall bei {tel_service} und {tkg_issue}",
            "Roaming/Entgelte bei {tel_service}",
            "SIM-Kartenmissbrauch/SIM-Swap und {tkg_issue}",
        ],
        slots={
            "tel_service": ["Mobilfunkvertrag", "Internetanschluss", "Kabel-Internet", "Festnetzvertrag"],
            "tkg_issue": ["Entschädigung", "Beschwerdeverfahren", "Schlichtung", "Rechnungseinspruch", "Sperre/Entsperre"],
        },
    )

    specs["ECG"] = LawSpec(
        templates=[
            "Impressumspflicht für {web}",
            "Haftung für Nutzerkommentare auf {web}",
            "E-Commerce: Informationspflichten bei {web}",
            "Spam/Newsletter: Einwilligung und Abmeldung bei {web}",
        ],
        slots={
            "web": ["Online-Shop", "Blog", "Social-Media-Seite", "Marktplatz-Profil", "Vereinswebsite"],
        },
    )

    specs["BAO"] = LawSpec(
        templates=[
            "Frist/Verspätung bei Steuererklärung und {bao_issue}",
            "Nachforderung/Säumniszuschlag und {bao_issue}",
            "Abgabenbescheid: {bao_remedy}",
            "Betriebsprüfung und Mitwirkungspflichten",
            "Stundung/Ratenzahlung von Abgaben und {bao_issue}",
        ],
        slots={
            "bao_issue": ["Zahlungserleichterung", "Nachsicht", "Zinsen", "Begründungspflicht", "Ratenplan"],
            "bao_remedy": ["Beschwerde", "Aussetzung der Einhebung", "Vorlageantrag", "Fristenlauf"],
        },
    )

    specs["AVG"] = LawSpec(
        templates=[
            "Verwaltungsverfahren: Parteiengehör bei {admin_subject}",
            "Akteneinsicht und {avg_issue}",
            "Bescheidzustellung und Fristenlauf bei {admin_subject}",
            "Säumnis der Behörde und {avg_issue}",
            "Wiedereinsetzung in den vorigen Stand wegen {avg_reason}",
        ],
        slots={
            "admin_subject": admin_subject,
            "avg_issue": ["Zuständigkeit", "Begründungspflicht", "Beweisaufnahme", "mündliche Verhandlung"],
            "avg_reason": ["Krankheit", "falscher Zustellung", "unverschuldetem Versäumnis", "Irrtum"],
        },
    )

    specs["VwGVG"] = LawSpec(
        templates=[
            "Beschwerde an das Landesverwaltungsgericht: {vwgv_detail}",
            "Aufschiebende Wirkung einer Beschwerde und {vwgv_detail}",
            "Mündliche Verhandlung vor dem Verwaltungsgericht und {vwgv_detail}",
            "Revision an den VwGH: Voraussetzungen und Fristen",
            "Erkenntnis/Beschluss: Zustellung und weitere Schritte ({vwgv_detail})",
        ],
        slots={
            "vwgv_detail": ["Fristen", "Formvorschriften", "Kosten", "Begründung", "Beweisanträge", "Parteistellung"],
        },
    )

    specs["VStG"] = LawSpec(
        templates=[
            "Verwaltungsstrafe wegen {traffic} und {vstg_issue}",
            "Strafverfügung/Anonymverfügung: Einspruch und {vstg_issue}",
            "Organstrafverfügung: {vstg_issue}",
            "Ersatzfreiheitsstrafe und {vstg_issue}",
            "Wiedereinsetzung/Fristversäumnis im Verwaltungsstrafverfahren",
        ],
        slots={
            "traffic": traffic,
            "vstg_issue": ["Fristen", "Zuständigkeit", "Beweislast", "Kosten", "Rechtsmittel", "Zahlungsfrist", "Akteneinsicht"],
        },
    )

    specs["StVO"] = LawSpec(
        templates=[
            "{traffic} in {traffic_context}: welche Verkehrsregel gilt?",
            "Pflichten im Straßenverkehr bei {traffic} in {traffic_context} und {stvo_issue}",
            "Parken/Halten: {parking_issue} in {traffic_context} und {stvo_issue}",
            "Verhalten nach Unfall: {accident_case} und {stvo_issue}",
        ],
        slots={
            "traffic": traffic,
            "traffic_context": ["30er-Zone", "Schulzone", "Autobahn", "Baustelle", "bei Nässe", "nachts", "während Probezeit", "bei Radfahrstreifen"],
            "parking_issue": ["Falschparken", "Halten im Halteverbot", "Parken vor Einfahrt", "Kurzparkzone und Parkschein", "Ladezone/Behindertenparkplatz"],
            "accident_case": ["Anhaltepflicht", "Absicherung/Warndreieck", "Daten austauschen", "Polizei verständigen", "Unfallflucht-Verdacht"],
            "stvo_issue": ["Tatbestand", "Ausnahmen", "Beweislage", "Sonderregeln (Probezeit)", "Rechtsfolgen"],
        },
    )

    specs["EKHG"] = LawSpec(
        templates=[
            "Haftung nach Verkehrsunfall zwischen {vehicle_a} und {vehicle_b}",
            "Halterhaftung vs Lenkerhaftung bei Unfall mit {vehicle_b}",
            "Mitverschulden und Schadenersatz bei Unfall mit {vehicle_b}",
            "Ansprüche von Mitfahrer:in nach Unfall mit {vehicle_a}",
        ],
        slots={
            "vehicle_a": ["PKW", "LKW", "Motorrad", "Bus", "Straßenbahn"],
            "vehicle_b": ["PKW", "Fahrrad", "Fußgänger:in", "E-Scooter", "Motorrad"],
        },
    )

    specs["VKG"] = LawSpec(
        templates=[
            "Väterkarenz nach {family_event}: Voraussetzungen und {vkg_issue}",
            "Meldung der Väterkarenz beim Arbeitgeber: {notice_detail} und {vkg_issue}",
            "Kündigungs-/Entlassungsschutz in der Väterkarenz und {vkg_issue}",
            "Papamonat/Familienzeit nach {family_event}: Fristen und {vkg_issue}",
            "Änderung eines Karenzteils: {change_detail} und {vkg_issue}",
            "Teilung der Karenz zwischen Elternteilen: {split_detail} und {vkg_issue}",
        ],
        slots={
            "family_event": ["Geburt", "Adoption", "Pflegekindaufnahme"],
            "notice_detail": [
                "Bekanntgabe von Beginn und Dauer",
                "8-Wochen-Frist nach der Geburt",
                "3-Monats-Vorankündigung",
                "Schriftform und Nachweis",
                "Änderungsmeldung/Verlängerungsmeldung",
            ],
            "change_detail": ["Verlängerung", "vorzeitiges Ende", "Wechsel des Elternteils", "Verschiebung des Antritts"],
            "split_detail": [
                "abwechselnde Karenz",
                "Karenz in mehreren Teilen",
                "Karenz im Anschluss an Mutterschutz",
                "gleichzeitige Karenzteile",
            ],
            "vkg_issue": [
                "Anspruchsvoraussetzungen",
                "Dauer/Antritt",
                "Fristen",
                "Nachweise",
                "Kündigungsschutz",
                "Entlassungsschutz",
                "Rückkehrrecht",
                "Teilzeit im Anschluss",
                "Sonderfälle (Früh-/Mehrlingsgeburt)",
            ],
        },
    )

    specs["SH-GG"] = LawSpec(
        templates=[
            "Sozialhilfe: Anspruchsvoraussetzungen und {sh_detail}",
            "Anrechnung von Einkommen/Vermögen in der Sozialhilfe und {sh_detail}",
            "Haushaltsgemeinschaft und Leistungshöhe: {sh_detail}",
            "Geld- vs Sachleistung bei Wohnkosten und {sh_detail}",
            "Härtefallklausel/Nichtanrechnung bestimmter Leistungen und {sh_detail}",
        ],
        slots={
            "sh_detail": [
                "Wohnkostenpauschale", "Mitwirkungspflichten", "Nachweispflichten", "Fristen", "Rückforderung",
                "Nichtanrechnung von Schmerzensgeld", "Unterstützung für Alleinerziehende", "Zuständigkeit des Landes",
            ],
        },
    )

    specs["StGB"] = LawSpec(
        templates=[
            "Betrug im Zusammenhang mit {fraud_channel}",
            "Körperverletzung nach {incident}",
            "Sachbeschädigung an {good}",
            "Diebstahl/Unterschlagung von {good}",
            "Nötigung/Bedrohung in {context}",
            "Cybercrime: {cyber}",
        ],
        slots={
            "fraud_channel": ["Online-Kleinanzeigen", "Online-Shop", "Ticketverkauf", "Kryptozahlung", "Telefonanruf"],
            "incident": ["Auseinandersetzung", "Verkehrsstreit", "Party", "Sportveranstaltung"],
            "good": goods,
            "context": ["Nachbarschaftskonflikt", "Arbeitsplatz", "öffentlichem Raum", "Online-Chat"],
            "cyber": ["Phishing", "Account-Übernahme", "Identitätsdiebstahl", "Datenmanipulation", "Ransomware"],
        },
    )

    specs["StPO"] = LawSpec(
        templates=[
            "Vorladung als Beschuldigte:r: Rechte und {stpo_issue}",
            "Hausdurchsuchung: Voraussetzungen und {stpo_issue}",
            "Akteneinsicht/Verteidigung: {stpo_issue}",
            "U-Haft: Voraussetzungen, Haftprüfung und {stpo_issue}",
            "Diversion/Einstellung: {stpo_issue}",
        ],
        slots={
            "stpo_issue": ["Rechtsmittel", "Fristen", "Belehrung", "Beweisverwertung", "Anwalt", "Protokollierung", "Akteneinsicht"],
        },
    )

    specs["ZPO"] = LawSpec(
        templates=[
            "Zivilprozess: Klage wegen {claim} und {zpo_issue}",
            "Mahnverfahren/Exekutionstitel: {zpo_issue}",
            "Beweisaufnahme (Zeug:innen, Gutachten) und {zpo_issue}",
            "Kostenrisiko/Prozesskostenersatz bei {claim}",
            "Vergleich und Klagsrückziehung: {zpo_issue}",
        ],
        slots={
            "claim": ["Zahlungsverzug", "Schadenersatz", "Gewährleistung", "Unterlassung", "Kautionsrückzahlung", "Werklohn"],
            "zpo_issue": ["Zuständigkeit", "Fristen", "Klagebeantwortung", "Beweisantrag", "Vergleich", "Berufung", "Kosten"],
        },
    )

    # ------------------------------------------------------------------
    # Additional major Austrian laws (expanded coverage)
    # ------------------------------------------------------------------
    # Topics are short, user-like issue phrases. Facets are added later automatically.
    GENERIC_TOPICS_BY_LAW: Dict[str, List[str]] = {
        "APG": [
            "Pensionskonto: Kontogutschriften prüfen",
            "Korridorpension: Voraussetzungen",
            "Alterspension: Antrittsalter",
            "Wartezeiten und Versicherungsmonate",
            "Ausgleichszulage (Abgrenzung)",
            "Pensionsantrag: Fristen und Unterlagen",
            "Nachkauf von Versicherungszeiten",
            "Anrechnung Kindererziehungszeiten",
            "Pensionshöhe berechnen",
            "Beschwerde gegen Pensionsbescheid",
            "Rehabilitation und Invalidität",
            "Teilpension/Zuverdienst",
        ],
        "AVRAG": [
            "Dienstzettel: Pflichtangaben und Frist",
            "Betriebsübergang: Übergang von Arbeitsverhältnissen",
            "Informationspflicht bei Betriebsübergang",
            "Kündigung wegen Betriebsübergang",
            "Entsendung nach Österreich: Arbeitsbedingungen",
            "Lohn- und Sozialdumping: Nachweise",
            "Haftung bei Unterentlohnung",
            "Rechte der Arbeitnehmer:innen beim Übergang",
            "Zuständigkeit Arbeits- und Sozialgericht",
            "Fristen für Ansprüche",
            "Beweisführung und Dokumente",
            "Schadenersatz bei Verstoß",
        ],
        "AWG": [
            "Entsorgungspflicht für gefährliche Abfälle",
            "Altstoffsammelzentrum: Abgabe von Problemstoffen",
            "Abfalltrennung in Mehrparteienhaus",
            "Abfallbeauftragte:r und Pflichten im Betrieb",
            "Abfallbilanz und Dokumentationspflicht",
            "Illegale Ablagerung und Verwaltungsstrafe",
            "Transport von Abfällen: Begleitschein",
            "Deponie und Genehmigung",
            "Rücknahme von Elektroaltgeräten",
            "Abfallende/End-of-Waste",
            "Kontrollen durch Behörde",
            "Zuständigkeit und Anzeige",
        ],
        "AZG": [
            "Arbeitszeitaufzeichnungen: Pflichten",
            "Überstunden: Zuschläge und Abgeltung",
            "Ruhezeiten und Ruhepausen",
            "Gleitzeitvereinbarung und Grenzen",
            "Höchstarbeitszeit und Durchrechnungszeitraum",
            "Sonn- und Feiertagsarbeit: Ausnahmen",
            "Bereitschaftsdienst und Rufbereitschaft",
            "Verstöße und Verwaltungsstrafen",
            "Kontrolle durch Arbeitsinspektorat",
            "Arbeitszeit bei Teilzeit",
            "Rechtsdurchsetzung und Nachweise",
            "Betriebsvereinbarung zur Arbeitszeit",
        ],
        "AlVG": [
            "Anspruch auf Arbeitslosengeld: Voraussetzungen",
            "Sperrfrist nach Kündigung/Entlassung",
            "Notstandshilfe: Anspruch und Höhe",
            "Meldepflicht beim AMS und Termine",
            "Zumutbarkeit von Jobs und Sanktionen",
            "Nebenverdienst und Anrechnung",
            "Krankenstand während Arbeitslosigkeit",
            "Auslandsaufenthalt und Leistungsbezug",
            "Rückforderung von Leistungen",
            "Rechtsmittel gegen AMS-Bescheid",
            "Schulung/Weiterbildung und Pflichten",
            "Beweismittel und Fristen",
        ],
        "AngG": [
            "Kündigungsfristen für Angestellte",
            "Entlassung: Gründe und Anfechtung",
            "Provisionsanspruch nach Beendigung",
            "Dienstverhinderung und Entgeltfortzahlung",
            "Konkurrenzklausel und Karenzentschädigung",
            "Abfertigung alt/neu (Abgrenzung)",
            "Überstundenpauschale und Grenzen",
            "Urlaubsansprüche (Abgrenzung UrlG)",
            "Zeugnisanspruch",
            "Rückzahlung von Ausbildungskosten",
            "Schadenersatzpflichten",
            "Zuständigkeit Arbeitsgericht",
        ],
        "AsylG": [
            "Asylantrag stellen: Ablauf und Zuständigkeit",
            "Subsidiärer Schutz und Verlängerung",
            "Dublin-Verfahren und Überstellung",
            "Beschwerde gegen negativen Asylbescheid",
            "Aufenthaltsrecht während des Asylverfahrens",
            "Familienverfahren im Asylrecht",
            "Sicherer Herkunftsstaat und Folgen",
            "Mitwirkungspflichten und Identitätsdokumente",
            "Aberkennung Schutzstatus",
            "Grundversorgung während des Verfahrens",
            "Fristen im BVwG-Verfahren",
            "Recht auf Rechtsberatung",
        ],
        "AußStrG": [
            "Pflegschaftsverfahren (Obsorge/Unterhalt)",
            "Erwachsenenvertretung und Genehmigungen",
            "Verlassenschaft/Erbrechtliche Abhandlung",
            "Rekurs gegen Beschluss im Außerstreit",
            "Antrag auf einstweilige Verfügung im Pflegschaftsverfahren",
            "Kostenersatz im Außerstreitverfahren",
            "Parteistellung und Beteiligtenrechte",
            "Verfahrenshilfe im Außerstreit",
            "Genehmigung von Rechtsgeschäften Minderjähriger",
            "Änderungsantrag bei Obsorge/Unterhalt",
            "Fristen für Rekurs/Revisionsrekurs",
            "Zuständigkeit Bezirksgericht (Wohnsitz)",
        ],
        "AÜG": [
            "Equal Pay bei Leiharbeit",
            "Überlassungsvertrag und Informationspflicht",
            "Zulässige Dauer der Überlassung",
            "Kündigung/Beendigung bei Beschäftiger",
            "Haftung für Entgelt",
            "Arbeitszeit und Überstunden bei Überlassung",
            "Betriebsrat und Mitbestimmung",
            "Beschwerde bei Verstößen",
            "Sanktionen und Verwaltungsstrafen",
            "Unterkunft/Spesen bei Einsatz",
            "Schutzvorschriften für Leiharbeitnehmer:innen",
            "Zuständigkeit Arbeitsinspektorat",
        ],
        "B-VG": [
            "Gleichheitsgrundsatz und sachliche Rechtfertigung",
            "Verhältnismäßigkeit bei Eingriffen in Grundrechte",
            "Kompetenzverteilung Bund/Land bei Gesetzgebung",
            "Gesetzesvorbehalt und Bestimmtheitsgebot",
            "Grundrecht auf Datenschutz im Verfassungsrang",
            "Freiheit der Meinungsäußerung und ihre Schranken",
            "Eigentumsschutz bei Enteignung/Entschädigung",
            "Legalitätsprinzip im Verwaltungsverfahren",
            "Objektivität/Sachlichkeit staatlichen Handelns",
            "Recht auf ein faires Verfahren (verfassunglich)",
            "Gleichbehandlung bei Förderungen/Beihilfen",
            "Rückwirkungsverbot und Vertrauensschutz",
        ],
        "BEinstG": [
            "Ausgleichstaxe berechnen und zahlen",
            "Begünstigte Behinderte: Feststellung",
            "Kündigungsschutz und Zustimmung",
            "Einstellungspflicht und Quoten",
            "Förderungen für Arbeitgeber:innen",
            "Behindertenpass und Nachweise",
            "Diskriminierung am Arbeitsplatz (Abgrenzung GlBG)",
            "Schlichtungsverfahren und Ansprüche",
            "Zuständigkeit Sozialministeriumservice",
            "Strafen bei Verstoß gegen Pflichten",
            "Arbeitsplatzadaptierung",
            "Fristen und Verfahren",
        ],
        "BWG": [
            "Bankgeheimnis: Auskunft an Dritte?",
            "Kontoeröffnung und KYC-Nachweise",
            "Kontosperre und Verdachtsmeldung",
            "Kreditkündigung und Fälligstellung",
            "Überziehungszinsen und Gebühren",
            "Beschwerde bei Bank/Ombudsstelle",
            "FMA-Beschwerde und Aufsicht",
            "Haftung der Bank bei Überweisung",
            "SEPA-Überweisung: Rückruf und Fristen",
            "Sicherheiten und Pfandrecht",
            "Verbraucherkredit (Abgrenzung VKrG)",
            "Zahlungskonto: Basiskonto",
        ],
        "DSGVO": [
            "Auskunft über verarbeitete personenbezogene Daten",
            "Löschung/Recht auf Vergessenwerden",
            "Widerspruch gegen Direktmarketing/Profiling",
            "Berichtigung falscher Daten",
            "Datenübertragbarkeit zwischen Diensten",
            "Einwilligung widerrufen und Folgen",
            "Auftragsverarbeitung und AV-Vertrag",
            "Datenschutz-Folgenabschätzung (DSFA) erforderlich?",
            "Meldung einer Datenpanne und Fristen",
            "Videoüberwachung am Arbeitsplatz/Mehrparteienhaus",
            "Drittlandtransfer und Standardvertragsklauseln",
            "Beschwerde bei der Datenschutzbehörde",
            "Speicherdauer und Löschkonzept",
            "Rechtsgrundlage: Vertrag vs. berechtigtes Interesse",
        ],
        "EO": [
            "Lohnpfändung/Exekution auf Arbeitseinkommen",
            "Fahrnisexekution in der Wohnung",
            "Exekution auf Bankguthaben",
            "Zwangsversteigerung einer Liegenschaft",
            "Sicherungsexekution vor Rechtskraft",
            "Exekutionstitel: Urteil, Vergleich, Zahlungsbefehl",
            "Einstellung oder Aufschub der Exekution",
            "Exekutionsbewilligung und Zuständigkeit",
            "Rechtsmittel im Exekutionsverfahren",
            "Pfändungsfreigrenzen und Existenzminimum",
            "Drittschuldnererklärung und Pfändungsprotokoll",
            "Rangordnung mehrerer Gläubiger:innen",
            "Kosten der Exekution und Kostenvorschuss",
        ],
        "EStG": [
            "Arbeitnehmerveranlagung: Werbungskosten geltend machen",
            "Pendlerpauschale und Pendlereuro",
            "Homeoffice-Pauschale und Arbeitsmittel",
            "Außergewöhnliche Belastung (Krankheitskosten)",
            "Sonderausgaben (Versicherungen/Spenden)",
            "Familienbonus Plus und Kindermehrbetrag",
            "Steuerpflicht bei Nebenjob/geringfügiger Beschäftigung",
            "Einkünfte aus Vermietung und Verpachtung",
            "Betriebsausgaben bei Selbständigkeit",
            "Abschreibung (AfA) für Arbeitsmittel/Anlagevermögen",
            "Reisekosten/Taggelder bei Dienstreise",
            "Doppelte Haushaltsführung und Familienheimfahrten",
            "Steuerfreie Zuschläge/Überstunden",
            "Kapitalerträge und KESt-Abzug",
            "Verlustausgleich und Verlustvortrag",
            "Sachbezug bei Dienstwagen",
            "Steuererklärung: Frist und elektronische Abgabe",
            "Rückzahlung/Steuernachzahlung nach Veranlagung",
            "Unterhaltsabsetzbetrag und Alleinverdiener:innen",
            "Progression bei Nachzahlungen",
            "Privatanteil bei gemischt genutzten Wirtschaftsgütern",
            "Grenzgänger:innen und Doppelbesteuerung",
        ],
        "EheG": [
            "einvernehmliche Scheidung: Voraussetzungen und Ablauf",
            "Scheidung aus Verschulden: Beweis und Rechtsfolgen",
            "Unterhalt nach Scheidung (Ehegattenunterhalt)",
            "Aufteilung eheliches Gebrauchsvermögen und Ersparnisse",
            "Ehewohnung: Benützungsregelung nach Scheidung",
            "Kontaktrecht/Obsorge im Zusammenhang mit Scheidung",
            "Trennungsjahr und unheilbare Zerrüttung",
            "Scheidungsfolgenvereinbarung",
            "Scheidung trotz Widerspruch",
            "Eheliche Aufteilung bei gemeinsamem Kredit",
            "Kosten der Scheidung und Verfahrenshilfe",
            "Vereinbarung über Unterhalt und Indexanpassung",
        ],
        "ElWOG": [
            "Lieferantenwechsel Strom: Fristen",
            "Netzzugang und Netzanschluss",
            "Netzentgelte und Abrechnung",
            "Grundversorgung bei Zahlungsproblemen",
            "Stromabschaltung: Voraussetzungen",
            "Messung/Zähler und Smart Meter",
            "Schlichtung bei Streit mit Stromanbieter",
            "Einspeisung Photovoltaik und Netz",
            "Vertrag kündigen und Endabrechnung",
            "Zuständigkeit E-Control",
            "Beschwerde über Netzbetreiber",
            "Pflichten bei Stromausfall",
        ],
        "FPG": [
            "Rückkehrentscheidung und Frist zur Ausreise",
            "Einreiseverbot: Dauer und Aufhebung",
            "Schubhaft: Voraussetzungen und Rechtsschutz",
            "Abschiebung und Duldung",
            "Mitwirkungspflichten bei Identitätsklärung",
            "Verhältnismäßigkeit bei Rückkehrmaßnahmen",
            "Beschwerde gegen FPG-Bescheid",
            "Aufenthaltsbeendigung trotz Familie",
            "Ausweisung nach Verwaltungsübertretung",
            "Rechtsberatung und Verfahrenshilfe",
            "Anordnung gelinderer Mittel statt Schubhaft",
            "Dokumente/Reisedokumente und Beschaffung",
        ],
        "FSG": [
            "Entzug der Lenkberechtigung nach Alkohol/Drogen",
            "Nachschulung und verkehrspsychologische Untersuchung",
            "Vormerksystem und Vormerkdelikte",
            "Probezeit-Führerschein und Verlängerung",
            "Medizinisches Gutachten zur Fahrtauglichkeit",
            "Wiedererteilung nach Entzug",
            "Führerschein umschreiben (Ausland)",
            "Lenkberechtigungsklassen und Berechtigungen",
            "Beschwerde gegen Entziehungsbescheid",
            "Sofortmaßnahmen der Behörde",
            "Fristen und Zuständigkeit (BH/Magistrat)",
            "Punkte und Maßnahmenstufen",
        ],
        "FinStrG": [
            "Selbstanzeige: Voraussetzungen und Wirkung",
            "Abgabenhinterziehung: Tatbestand und Vorsatz",
            "Finanzstrafverfahren: Einleitung und Rechte",
            "Beschlagnahme/Sicherstellung im Finanzstrafverfahren",
            "Geldstrafe und Ersatzfreiheitsstrafe",
            "Absehen von Strafe (Diversion) im Finanzstrafrecht",
            "Verjährung im Finanzstrafverfahren",
            "Zollvergehen und Einfuhrabgaben",
            "Beschwerde gegen Finanzstrafbescheid",
            "Hausdurchsuchung im Finanzstrafverfahren (Abgrenzung)",
            "Ratenzahlung/Zahlungsaufschub bei Geldstrafe",
            "Mitwirkungspflichten und Aussageverweigerung",
        ],
        "GSVG": [
            "SVS-Pflichtversicherung für Selbständige",
            "Beitragsgrundlage und Mindestbeiträge",
            "Nachbemessung nach Einkommensteuerbescheid",
            "Krankenversicherung: Leistungen und Anspruch",
            "Pensionsversicherung: Beitragszeiten",
            "Unfallversicherung und Arbeitsunfall",
            "Ratenzahlung/Stundung von Beiträgen",
            "Meldepflichten bei Beginn/Ende der Selbständigkeit",
            "Selbstversicherung und freiwillige Weiterversicherung",
            "Beschwerde gegen SVS-Bescheid",
            "Säumniszuschlag und Mahnverfahren",
            "Mitversicherung von Angehörigen",
        ],
        "GewO": [
            "Gewerbeanmeldung für ein freies Gewerbe",
            "Befähigungsnachweis bei reglementiertem Gewerbe",
            "Gastgewerbeberechtigung und Auflagen",
            "Betriebsanlagenbewilligung und Nachbarn",
            "Gewerberechtlicher Geschäftsführer: Bestellung",
            "Gewerbeausübung ohne Berechtigung: Strafe",
            "Ruhendmeldung und Wiederaufnahme",
            "Standortverlegung und Meldepflicht",
            "Überprüfung durch Behörde (Kontrolle)",
            "Gewerbeentziehung wegen Unzuverlässigkeit",
            "Nebenrechte und Hilfsgewerbe",
            "Gewerbe in Wohnung/Homeoffice: Zulässigkeit",
        ],
        "GrEStG": [
            "Grunderwerbsteuer beim Kauf einer Wohnung",
            "Grunderwerbsteuer bei Schenkung/Erbschaft",
            "Bemessungsgrundlage und Gegenleistung",
            "Befreiungen (z.B. Familienkreis) und Voraussetzungen",
            "Selbstberechnung durch Notar:in/RA",
            "Anzeigepflichten und Fristen",
            "Anteilsvereinigung/Share Deal und GrESt",
            "Rückabwicklung des Kaufvertrags und Erstattung",
            "GrESt bei Übergabe im Rahmen Scheidung",
            "Stundung/Ratenzahlung von GrESt",
            "GrESt bei Grundstückstausch",
            "Rechtsmittel gegen GrESt-Bescheid",
        ],
        "IG-L": [
            "Fahrverbot wegen Luftreinhalteplan",
            "Ausnahmebewilligung für Anrainer/Handwerk",
            "Kontrolle und Strafen bei Fahrverbot",
            "Umweltzone und Kennzeichnung",
            "Feinstaub/NO2 Grenzwerte und Maßnahmen",
            "Einwendungen gegen Verordnung",
            "Transport und Lieferverkehr: Ausnahmen",
            "Gültigkeit und Umfang der Maßnahme",
            "Beschwerde gegen Strafe",
            "Zuständigkeit BH/Magistrat",
            "Nachweis der Ausnahme",
            "Temporäre Maßnahmen bei Smog",
        ],
        "IO": [
            "Privatinsolvenz/Schuldenregulierung",
            "Sanierungsverfahren mit/ohne Eigenverwaltung",
            "Insolvenzantrag: Fristen und Voraussetzungen",
            "Forderungsanmeldung im Insolvenzverfahren",
            "Anfechtung von Rechtshandlungen vor Insolvenz",
            "Zahlungsplan: Annahme und Mindestquote",
            "Abschöpfungsverfahren und Restschuldbefreiung",
            "Sperre von Konten und Zahlungsverkehr",
            "Rechte der Arbeitnehmer:innen bei Insolvenz",
            "Insolvenzverwalter: Aufgaben und Berichtspflichten",
            "Behandlung gesicherter Forderungen",
            "Fortführung vs. Schließung des Betriebs",
        ],
        "KAKuG": [
            "Aufnahme in die Krankenanstalt",
            "Pflegegebühren und Kostenersatz",
            "Entlassungsmanagement und Information",
            "Kostentragung durch Versicherung/Patient:in",
            "Wahlärztliche Leistungen im Spital",
            "Patientenanwaltschaft einschalten",
            "Dokumentation und Einsicht in Krankengeschichte",
            "Haftung bei Behandlungsfehler (Abgrenzung ABGB)",
            "Aufklärungspflichten (Abgrenzung)",
            "Beschwerde über Spitalsbehandlung",
            "Zuständigkeit und Verfahren",
            "Hausordnung und Besuchsrechte",
        ],
        "KFG": [
            "Zulassung eines Fahrzeugs und erforderliche Unterlagen",
            "Pickerl (§57a) und Mängel",
            "Umbau/Tuning: Genehmigung und Eintragung",
            "Halterpflichten und Inbetriebnahme",
            "Kennzeichenverlust und Ersatzkennzeichen",
            "Lenkerauskunft und Fristen",
            "Überladung und technische Vorschriften",
            "Fahrzeugstilllegung bei Gefahr im Verzug",
            "Versicherung und Zulassung",
            "Beschwerde gegen KFG-Bescheid",
            "Kurzzeitkennzeichen/Überstellungsfahrt",
            "Abgas-/Lärmvorschriften",
        ],
        "KStG": [
            "Mindest-KöSt und Fälligkeit",
            "verdeckte Gewinnausschüttung",
            "Gruppenbesteuerung: Voraussetzungen",
            "Abzugsfähigkeit von Betriebsausgaben",
            "Zinsschranken und Finanzierungskosten",
            "Gewinnausschüttung und KESt",
            "Gemeinnützigkeit und Körperschaftsteuer",
            "Umgründung und steuerliche Folgen",
            "Verlustvortrag bei Körperschaften",
            "KöSt-Erklärung: Fristen",
            "Abgrenzung Betriebsausgabe/Privatentnahme",
            "Transfer Pricing bei verbundenen Unternehmen",
        ],
        "KartG": [
            "Kartellverbot und abgestimmte Verhaltensweisen",
            "Missbrauch marktbeherrschender Stellung",
            "Fusionskontrolle: Anmeldepflicht",
            "Hausdurchsuchung (Dawn Raid) und Rechte",
            "Bußgeldverfahren und Bemessung",
            "Kronzeugenregelung (Leniency)",
            "Kooperationen/Joint Venture zulässig?",
            "Preisbindungen im Vertrieb",
            "Marktabgrenzung und Marktanteile",
            "Rechtsmittel im Kartellverfahren",
            "Unterlassungsanspruch von Mitbewerber:innen",
            "Zuständigkeit Kartellgericht/BWB",
        ],
        "KommStG": [
            "Kommunalsteuer: Bemessungsgrundlage (Lohnsumme)",
            "Betriebsstätte: Zuordnung und Abgrenzung",
            "Haftung des Arbeitgebers für Kommunalsteuer",
            "Anmeldung und Abfuhr der Kommunalsteuer",
            "Kommunalsteuerprüfung und Nachforderung",
            "Kommunalsteuer bei Homeoffice/Remote Work",
            "Befreiungen/Erleichterungen",
            "Fristen und Säumniszuschläge",
            "Rechtsmittel bei Kommunalsteuerbescheid",
            "Kommunalsteuer bei Leiharbeit",
            "Kommunalsteuer bei Konzernstrukturen",
            "Zuständige Gemeinde und Aufteilung",
        ],
        "MeldeG": [
            "Hauptwohnsitz anmelden nach Umzug",
            "Nebenwohnsitz und Meldepflicht",
            "Meldezettel: Unterschrift Vermieter:in?",
            "Fristen für Anmeldung/Abmeldung",
            "Meldeauskunft: wer darf abfragen?",
            "Strafe bei verspäteter Meldung",
            "Meldepflicht bei Auszug ins Ausland",
            "Wohnsitzmeldung bei WG/Untermiete",
            "Meldebestätigung für Behörden",
            "Berichtigung falscher Meldedaten",
            "Meldung bei kurzfristiger Unterkunft",
            "Zuständigkeit Gemeinde/Magistrat",
        ],
        "NAG": [
            "Rot-Weiß-Rot Karte: Voraussetzungen",
            "Aufenthaltstitel für Studierende",
            "Familiennachzug: Nachweise und Fristen",
            "Verlängerung eines Aufenthaltstitels",
            "Daueraufenthalt - EU: Voraussetzungen",
            "Aufenthaltsbewilligung und Erwerbstätigkeit",
            "Ausreise/Aufenthaltsunterbrechung und Folgen",
            "Ablehnung eines NAG-Antrags: Rechtsmittel",
            "Quotenplatz und Antragstellung",
            "Zuständigkeit (BH/MA 35) und Termin",
            "Krankenversicherung/Nachweis Mittel",
            "Wohnraumnachweis und Mietvertrag",
        ],
        "NoVAG": [
            "NoVA bei Neuwagenkauf in Österreich",
            "NoVA beim Import eines Fahrzeugs aus der EU",
            "Befreiung/Ermäßigung bei E-Auto oder Sonderfällen",
            "Bemessungsgrundlage und Berechnung",
            "NoVA-Nachzahlung bei Änderungen",
            "NoVA bei Leasing",
            "Rückerstattung bei Export",
            "Fristen und Anzeige beim Finanzamt",
            "NoVA und Gebrauchtwagenhandel",
            "Strafen bei falscher NoVA-Angabe",
            "NoVA bei Motorrad/leichte Kfz",
            "Zusammenhang NoVA und CO2-Wert",
        ],
        "PSG": [
            "Privatstiftung gründen: Stiftungsurkunde",
            "Begünstigte und Begünstigtenkreis",
            "Stiftungsvorstand: Pflichten und Haftung",
            "Widerrufsvorbehalt und Auflösung",
            "Stiftungszweck ändern",
            "Rechnungslegung und Prüfung",
            "Auszahlung an Begünstigte: Voraussetzungen",
            "Stiftungszusatzurkunde",
            "Anfechtung von Stiftungsbeschlüssen",
            "Zuständigkeit Firmenbuchgericht",
            "Steuerliche Aspekte (Abgrenzung)",
            "Kosten und Gebühren",
        ],
        "PatG": [
            "Patent anmelden: Ablauf und Kosten",
            "Neuheit/Erfinderische Tätigkeit beurteilen",
            "Patentverletzung und Unterlassung",
            "Lizenzvertrag und Lizenzgebühren",
            "Nichtigkeitsverfahren gegen Patent",
            "Priorität und Anmeldung im Ausland",
            "Arbeitnehmererfindung und Rechte",
            "Schutzumfang und Patentansprüche",
            "Jahresgebühren und Aufrechterhaltung",
            "Rechtsmittel gegen Patentamt-Entscheidung",
            "Geheimhaltung vor Anmeldung",
            "Durchsetzung (Einstweilige Verfügung)",
        ],
        "SMG": [
            "Suchtmittelbesitz für Eigengebrauch: Folgen",
            "Diversion/Therapie statt Strafe",
            "Sicherstellung und Vernichtung von Suchtmitteln",
            "Verdacht auf Handel/Weitergabe",
            "Anzeige und Rechte bei Polizeikontrolle",
            "Hausdurchsuchung (Abgrenzung StPO)",
            "Strafrahmen und Milderungsgründe",
            "Rehabilitation/Eintrag Strafregister",
            "Substitutionsbehandlung und Auflagen",
            "Mengenbegriffe und Qualifikation",
            "Jugendliche im Suchtmittelverfahren",
            "Beschwerde gegen Maßnahmen",
        ],
        "SPG": [
            "Wegweisung und Betretungsverbot bei häuslicher Gewalt",
            "Identitätsfeststellung durch Polizei: Voraussetzungen",
            "Anhaltung/Anhaltung von Personen",
            "Durchsuchung von Personen und Sachen",
            "Betretungsverbot: Dauer und Verlängerung",
            "Gefährderansprache und Sicherstellung von Gegenständen",
            "Waffenverbot und Abnahme",
            "Gefährdungsprognose und Dokumentation",
            "Beschwerde gegen polizeiliche Maßnahme",
            "Schutz von gefährdeten Personen",
            "Videoüberwachung durch Sicherheitsbehörden",
            "Verhältnismäßigkeit polizeilicher Zwangsgewalt",
        ],
        "SchUG": [
            "Schularbeit: Beurteilung und Wiederholung",
            "Nicht genügend und Aufsteigen",
            "Wiederholungsprüfung: Anmeldung",
            "Ausschluss von Schüler:in",
            "Disziplinarmaßnahmen und Verfahren",
            "Entschuldigung/Beurlaubung",
            "Schulpflicht und Fernbleiben",
            "Rechtsmittel gegen Beurteilung",
            "Schulwechsel und Aufnahme",
            "Konflikte mit Lehrkraft: Schulleitung",
            "Leistungsfeststellung und Transparenz",
            "Zuständigkeit Bildungsdirektion",
        ],
        "StbG": [
            "Einbürgerung: Mindestaufenthalt und Voraussetzungen",
            "Staatsbürgerschaft durch Abstammung",
            "Doppelstaatsbürgerschaft: Zulässigkeit",
            "Entziehung/Verlust der Staatsbürgerschaft",
            "Nachweis Deutschkenntnisse und Integration",
            "Einbürgerung trotz Strafregister?",
            "Bescheid über Staatsbürgerschaft: Rechtsmittel",
            "Verleihung als Ermessensentscheidung",
            "Staatsbürgerschaft für Ehepartner:in",
            "Staatsbürgerschaft für Kinder",
            "Gebühren und Verfahrenskosten",
            "Fristen und Zuständigkeit (BH/Magistrat)",
        ],
        "TSchG": [
            "Hundehaltung: Pflichten und Verbote",
            "Tierquälerei melden: Zuständigkeit",
            "Haltungsbedingungen für Nutztiere",
            "Tierversuche: Bewilligung und Kontrollen",
            "Tiertransport: Anforderungen",
            "Verbot bestimmter Haltungsformen",
            "Sicherstellung von Tieren bei Missstand",
            "Strafen nach Tierschutzrecht",
            "Tierärztliche Auflagen und Amtstierarzt",
            "Privathaltung exotischer Tiere",
            "Kastrationspflicht bei Katzen (regional)",
            "Beschwerde gegen Tierschutzbescheid",
        ],
        "UG": [
            "Zulassung zu Studium und Fristen",
            "Studienbeitrag: Befreiung/Ermäßigung",
            "Prüfungswiederholung und Antritt",
            "Anerkennung von Prüfungen",
            "Beurlaubung und Unterbrechung",
            "Disziplinarverfahren an Universität",
            "Studienrechtliche Bescheide und Rechtsmittel",
            "Plagiat und Konsequenzen",
            "Inskription und Fortsetzungsbestätigung",
            "ÖH-Beitrag und Versicherung",
            "Studienplan und ECTS",
            "Zuständigkeit Studienpräses/Rektorat",
        ],
        "UStG": [
            "Vorsteuerabzug: Voraussetzungen und Nachweise",
            "Rechnungserfordernisse und Berichtigung",
            "Kleinunternehmerregelung und Umsatzgrenzen",
            "Reverse-Charge bei Bauleistungen/Leistungen aus dem Ausland",
            "innergemeinschaftliche Lieferung und UID-Prüfung",
            "innergemeinschaftlicher Erwerb und Zusammenfassende Meldung",
            "Umsatzsteuer-Voranmeldung (UVA): Frist und Zahlung",
            "Soll- vs. Ist-Besteuerung",
            "Steuersätze (Normal/ermäßigt) bei bestimmten Leistungen",
            "Gutschriftverfahren und Storno",
            "Unechte/echte Steuerbefreiungen",
            "Ort der Leistung bei Dienstleistungen",
            "Differenzbesteuerung (Gebrauchtwaren)",
            "Importumsatzsteuer und Zoll",
            "Privatanteil und Eigenverbrauch",
            "UID-Nummer beantragen/prüfen",
            "Innerhalb EU: Dreiecksgeschäft",
            "Vorsteuerkorrektur (Berichtigung) bei Änderung der Nutzung",
            "Elektronische Dienstleistungen und OSS",
        ],
        "UVP-G": [
            "UVP-Pflicht für Bauprojekt/Industrieanlage",
            "Screening und Schwellenwerte",
            "Bürger:innenbeteiligung und Einwendungen",
            "Umweltverträglichkeitsgutachten",
            "Genehmigungsverfahren und Fristen",
            "Beschwerde gegen UVP-Bescheid",
            "Zusammenwirken mit Naturschutzrecht",
            "Kumulierung von Projekten",
            "Standortprüfung und Alternativen",
            "Auflagen und Monitoring",
            "Zuständigkeit Landesregierung",
            "Parteistellung von NGOs",
        ],
        "UWG": [
            "Irreführende Werbung im Online-Shop",
            "Aggressive Geschäftspraktiken und Druck",
            "Vergleichende Werbung: Zulässigkeit",
            "Schleichwerbung/Influencer-Kennzeichnung",
            "Lockangebote und Preisangaben",
            "Unterlassungsklage durch Mitbewerber:in",
            "Einstweilige Verfügung im UWG",
            "Schadenersatz bei unlauterem Wettbewerb",
            "Abmahnung und Kostenersatz",
            "Verletzung von Marktverhaltensregeln",
            "Belästigende Werbung (Spam) und Abgrenzung",
            "Beweislast und Dokumentation",
        ],
        "UrhG": [
            "Urheberrecht an Foto/Video: wer ist Urheber:in?",
            "Nutzungsrechte übertragen: Vertrag",
            "Verwendung von Bildern auf Social Media",
            "Zitatrecht und Schranken",
            "Abmahnung wegen Urheberrechtsverletzung",
            "Plagiat und Nachweis der Schöpfungshöhe",
            "Streaming/Sharing: Haftung",
            "Lizenzen (Creative Commons) richtig nutzen",
            "Vergütung und Lizenzgebühren",
            "Bildnisse und Persönlichkeitsrechte (Abgrenzung)",
            "Software-Lizenz und Urheberrecht",
            "Unterlassung und Schadenersatz",
        ],
        "UrlG": [
            "Urlaubsanspruch: Berechnung bei Teilzeit",
            "Urlaubsverbrauch und Zustimmung",
            "Urlaubsersatzleistung bei Beendigung",
            "Verjährung von Urlaubsansprüchen",
            "Betriebsurlaub: Zulässigkeit",
            "Urlaubssperre in Spitzenzeiten",
            "Krankenstand während Urlaub",
            "Übertrag von Resturlaub",
            "Pflegefreistellung (Abgrenzung)",
            "Anspruch in Probezeit",
            "Urlaub bei Karenz",
            "Beweis (Urlaubsaufzeichnungen)",
        ],
        "VbVG": [
            "Verbandsverantwortlichkeit bei Straftaten von Mitarbeiter:innen",
            "Compliance-Maßnahmen zur Haftungsvermeidung",
            "Verbandsgeldbuße und Bemessung",
            "Zurechnung: Entscheidungsträger vs. Mitarbeiter:innen",
            "Diversion für Verbände",
            "Interne Untersuchung und Mitwirkung",
            "Auflagen (Compliance-Programm) im Urteil",
            "Vorwurf Organisationsverschulden",
            "Verjährung bei Verbandsverantwortlichkeit",
            "Strafverfahren gegen Verband und natürliche Personen",
            "Rechtsmittel im VbVG-Verfahren",
            "Selbstanzeige/Kooperation und Strafmilderung",
        ],
        "VerG": [
            "Verein gründen: Statuten und Anzeige",
            "Vereinsregistereintragung",
            "Vorstand: Bestellung und Haftung",
            "Mitgliederversammlung und Beschlüsse",
            "Auflösung des Vereins",
            "Vereinsbehörde: Prüfungen und Aufträge",
            "Vereinsinterne Streitigkeiten",
            "Rechnungslegung und Transparenz",
            "Änderung der Statuten",
            "Untersagung der Vereinstätigkeit",
            "Spenden sammeln im Verein",
            "Haftung des Vereins für Schäden",
        ],
        "VersG": [
            "Versammlung/Demo anmelden: Fristen und Inhalt",
            "Spontanversammlung: Zulässigkeit",
            "Untersagung einer Versammlung und Rechtsschutz",
            "Auflagen (Route, Zeit, Ordner:innen)",
            "Auflösung durch Polizei: Voraussetzungen",
            "Störung/Blockade und Folgen",
            "Gegendemonstration und Abgrenzung",
            "Vermummungsverbot und Ausnahmen",
            "Sicherheit und Verkehrsumleitungen",
            "Strafen bei Verstößen gegen Auflagen",
            "Dokumentation durch Polizei/Video",
            "Zuständigkeit BH/Magistrat",
        ],
        "VersVG": [
            "Rücktritt vom Versicherungsvertrag",
            "Anzeigepflichten vor Vertragsabschluss",
            "Obliegenheiten im Schadensfall",
            "Prämienverzug und Kündigung",
            "Deckungsablehnung durch Versicherung",
            "Verjährung von Versicherungsansprüchen",
            "Schadensmeldung und Fristen",
            "Mitversicherung und Risikoänderung",
            "Regress der Versicherung",
            "Beweislast bei Obliegenheitsverletzung",
            "Teil-/Totalschaden und Entschädigung",
            "Beschwerde/Ombudsstelle",
        ],
        "VfGG": [
            "Beschwerde an den Verfassungsgerichtshof",
            "Frist und Form der VfGH-Beschwerde",
            "Individualantrag auf Gesetzesprüfung",
            "Antrag auf Verordnungsprüfung",
            "Verfahrenshilfe beim VfGH",
            "Zurückweisung wegen Subsidiarität",
            "Verfassungsrechtlich gewährleistete Rechte",
            "Gleichheitswidrigkeit geltend machen",
            "Kosten und Eingabegebühr",
            "Aufschiebende Wirkung/Provisorische Maßnahmen",
            "Vertretungspflicht im VfGH-Verfahren",
            "Entscheidung: Aufhebung und Kundmachung",
        ],
        "VwGG": [
            "Revision an den Verwaltungsgerichtshof: Voraussetzungen",
            "Zulässigkeit der Revision und Begründung",
            "Frist zur Revision und Formanforderungen",
            "Aufschiebende Wirkung beantragen",
            "Revisionsbeantwortung und Verfahren",
            "Kosten und Pauschalgebühr",
            "Außerordentliche Revision",
            "Verfahrenshilfe beim VwGH",
            "Zurückweisung wegen Unzulässigkeit",
            "Vertretungspflicht (RA) im VwGH-Verfahren",
            "Bindungswirkung von Tatsachenfeststellungen",
            "Entscheidungsformen des VwGH",
        ],
        "WGG": [
            "Mietzins bei gemeinnütziger Bauvereinigung",
            "Finanzierungsbeitrag/Genossenschaftsanteil",
            "Betriebskostenabrechnung im WGG",
            "Kaution und Rückzahlung",
            "Befristung/Weitergabe von WGG-Wohnungen",
            "Wohnungszuweisung und Warteliste",
            "Kaufoption/Übernahme in Eigentum",
            "Erhaltungs- und Verbesserungsarbeiten",
            "Kündigung und Räumung",
            "Rechtsmittel/Schlichtungsstelle im WGG",
            "Indexanpassung des Entgelts",
            "Mängel und Gewährleistung in Neubau",
        ],
        "WRG": [
            "Wasserrechtliche Bewilligung für Brunnen",
            "Einleitung von Abwasser in Gewässer",
            "Wasserbenutzungsrecht und Umfang",
            "Hochwasserschutz und Maßnahmen",
            "Gewässerrandstreifen und Nutzung",
            "Wasserschutzgebiet und Auflagen",
            "Genehmigung für Uferverbau",
            "Zuständigkeit Wasserrechtsbehörde",
            "Fristen und Auflagen in Bewilligungsbescheid",
            "Rechtsmittel gegen Wasserrechtsbescheid",
            "Haftung bei Gewässerverunreinigung",
            "Stilllegung von Anlagen",
        ],
        "WaffG": [
            "Waffenbesitzkarte beantragen",
            "Waffenpass: Voraussetzungen",
            "Sichere Verwahrung von Schusswaffen",
            "Waffenverbot: Anordnung und Rechtsschutz",
            "Verlässlichkeit und Eignung",
            "Verbotene Waffen und Ausnahmen",
            "Überlassen/Verkauf von Waffen",
            "Meldepflichten bei Waffen",
            "Aufhebung eines Waffenverbots",
            "Kontrolle durch Behörde",
            "Strafen bei Verstößen",
            "Transport von Waffen",
        ],
        "ZustG": [
            "Hinterlegung eines Bescheids und Fristbeginn",
            "RSa/RSb-Zustellung und Zustellnachweis",
            "Elektronische Zustellung und Zustelladresse",
            "Zustellmängel und Heilung",
            "Abwesenheit/Urlaub: Auswirkungen auf Zustellfiktion",
            "Ersatzzustellung an Haushaltsangehörige",
            "Zustellvollmacht und Zustellbevollmächtigte:r",
            "Zustellung ins Ausland",
            "Nachweisprobleme bei Zustellung",
            "Wiedereinsetzung wegen Zustellproblemen",
            "Behördliche Zustellung per Post",
            "Zustellversuch und Abholfrist",
        ],
    }

    def _add_topic_spec(law: str, topics: List[str]) -> None:
        specs[law] = LawSpec(templates=["{topic}"], slots={"topic": topics})

    for _law, _topics in GENERIC_TOPICS_BY_LAW.items():
        if _law not in specs:
            _add_topic_spec(_law, _topics)

    return specs

def estimate_max_issues(spec: LawSpec) -> int:
    """Upper bound estimate used for choosing a reasonable core-seed count."""
    total = 0
    for tmpl in spec.templates:
        keys = _PLACEHOLDER_RE.findall(tmpl)
        if not keys:
            total += 1
        else:
            prod = 1
            for k in keys:
                prod *= max(1, len(spec.slots.get(k, [])))
            total += prod
    return total

def generate_issues_for_law(law: str, spec: LawSpec, *, min_count: int, seed: int) -> List[str]:
    """Deterministically generate >= min_count distinct issue strings, or raise if impossible."""
    rng = random.Random((seed + stable_int(f"{law}:issues")) & 0xFFFFFFFF)

    issues: List[str] = []
    seen: Set[str] = set()

    templates = spec.templates[:]
    rng.shuffle(templates)

    # Bounded sampling per template (keeps runtime predictable)
    per_template_budget = max(60, ceil(min_count / max(1, len(templates))) * 10)

    # Pass-based loop to detect stagnation (prevents infinite loops)
    max_passes = 40
    for _pass in range(max_passes):
        before_pass = len(issues)
        for tmpl in templates:
            keys = _PLACEHOLDER_RE.findall(tmpl)
            if not keys:
                cand = normalize_ws(tmpl)
                if cand not in seen:
                    seen.add(cand)
                    issues.append(cand)
            else:
                slots = [spec.slots[k] for k in keys]
                used_sig: Set[Tuple[int, ...]] = set()
                for _ in range(per_template_budget):
                    sig = tuple(rng.randrange(len(s)) for s in slots)
                    if sig in used_sig:
                        continue
                    used_sig.add(sig)
                    mapping = {k: spec.slots[k][sig[i]] for i, k in enumerate(keys)}
                    cand = normalize_ws(tmpl.format(**mapping))
                    if cand not in seen:
                        seen.add(cand)
                        issues.append(cand)
                    if len(issues) >= min_count:
                        break
            if len(issues) >= min_count:
                break
        if len(issues) >= min_count:
            break
        if len(issues) == before_pass:
            break  # stagnation

    if len(issues) < min_count:
        cap = estimate_max_issues(spec)
        raise RuntimeError(
            f"Unable to generate enough issues for {law}: needed {min_count}, got {len(issues)} (estimated cap ~{cap})."
        )

    rng.shuffle(issues)
    return issues[:]

def expand_spec_with_facets(law: str, base: LawSpec, *, seed: int) -> LawSpec:
    """Automatic expansion: combine representative core issues with common legal facets."""
    base_cap = estimate_max_issues(base)
    core_seed = min(25, max(8, min(base_cap, 25)))  # robust even for small base caps

    # Generate representative cores from the base spec (low count)
    core_issues = generate_issues_for_law(law, base, min_count=min(core_seed, base_cap), seed=seed)

    spec = copy.deepcopy(base)
    spec.slots["core"] = core_issues
    spec.slots["facet"] = FACETS_COMMON

    # Facet expansion templates; yields hundreds of distinct, meaningful issues
    spec.templates.extend(
        [
            "{core}: {facet}",
            "{facet} zu {core}",
            "{core} - {facet}",
            "Frage zu {core}: {facet}",
        ]
    )
    return spec

# -------------------------
# Context and query generation
# -------------------------

def topic_context(topic_id: str, law: str, seed: int, law_context_prob: float) -> Dict[str, str]:
    rng = random.Random((seed + stable_int(f"{topic_id}:{law}:ctx")) & 0xFFFFFFFF)

    # Base pools
    actor_pool = ACTORS
    counterparty_pool = COUNTERPARTIES
    authority_pool = AUTHORITIES

    # Optional law-specific overrides (improves topical realism and discriminative signal).
    ov = LAW_CONTEXT_OVERRIDES.get(law, {})
    if ov and rng.random() < law_context_prob:
        actor_pool = ov.get("actors", actor_pool)
        counterparty_pool = ov.get("counterparties", counterparty_pool)
        authority_pool = ov.get("authorities", authority_pool)

    actor = rng.choice(actor_pool)
    counterparty = rng.choice(counterparty_pool)
    city = rng.choice(CITIES_AT)
    timep = rng.choice(TIME_PHRASES)
    amount = rng.choice(AMOUNTS)
    channel = rng.choice(CHANNELS)
    evidence = rng.choice(EVIDENCE)
    authority = rng.choice(authority_pool)

    return {
        "actor": actor,
        "counterparty": counterparty,
        "city": city,
        "time": timep,
        "amount": f"{amount} EUR",
        "amount_kw": f"{amount}eur",
        "channel": channel,
        "evidence": evidence,
        "authority": authority,
    }

def build_scenario(issue: str, ctx: Dict[str, str], rng: random.Random, term: str | None = None) -> str:
    # Vary surface forms and omit some fields to avoid a single dominant boilerplate pattern.
    amount_part = rng.choice([f"Es geht um {ctx['amount']}.", f"Betrag: {ctx['amount']}.", ""])
    contact_part = rng.choice([f"Kontakt über {ctx['channel']}.", f"Kommunikation via {ctx['channel']}.", ""])
    evidence_part = rng.choice(
        [
            f"Nachweis: {ctx['evidence']}.",
            f"Belege: {ctx['evidence']}.",
            f"Dokumentiert durch {ctx['evidence']}.",
            "",
        ]
    )
    authority_part = rng.choice(
        [
            f"Es liegt ein Schreiben/Bescheid von {ctx['authority']} vor.",
            f"Unklar ist die Zuständigkeit ({ctx['authority']} oder Gericht).",
            "",
        ]
    )
    term_part = ""
    if term and rng.random() < 0.55:
        term_part = rng.choice([f"Stichwort: {term}.", f"Thema: {term}.", f"({term})"])

    skeletons = [
        f"Ich bin {ctx['actor']} in {ctx['city']}. {ctx['time']} gab es mit {ctx['counterparty']} ein Problem: {issue}. {amount_part} {contact_part} {evidence_part} {authority_part} {term_part}",
        f"Sachverhalt ({ctx['city']}, {ctx['time']}): {issue}. Beteiligte: {ctx['actor']} vs {ctx['counterparty']}. {amount_part} {evidence_part} {term_part} {authority_part}",
        f"Kurzfall: {issue} - {ctx['actor']} ({ctx['city']}) vs {ctx['counterparty']}. {contact_part} {amount_part} {term_part}",
        f"{ctx['time']} in {ctx['city']}: {issue}. {evidence_part} {authority_part} {term_part}",
    ]
    return normalize_ws(rng.choice(skeletons))

STYLES = ["nl_short", "nl_long", "scenario", "procedural", "authority", "keyword", "fragment"]

STYLE_TEMPLATES: Dict[str, List[str]] = {
    "nl_short": [
        "{issue} - was gilt?",
        "Welche Rechte/Pflichten habe ich bei {issue}?",
        "Was kann ich bei {issue} tun?",
        "{issue}: welche Frist gilt?",
        "{issue}: was sind die Voraussetzungen?",
        "{issue} - brauche ich einen Antrag/Nachweis?",
        "{issue}: Behörde oder Gericht?",
        "Gibt es bei {issue} Ausnahmen?",
        "Welche Kosten/Risiken gibt es bei {issue}?",
        "Welche Sanktionen drohen bei {issue}?",
    ],
    "nl_long": [
        "{scenario} Welche Rechtslage gilt und welche Schritte sind sinnvoll?",
        "{scenario} Welche Ansprüche/Rechtsfolgen kommen in Betracht und welche Fristen laufen?",
        "{scenario} Welche Voraussetzungen sind relevant, welche Nachweise brauche ich und wo ist einzubringen?",
        "{scenario} Wie gehe ich praktisch vor (Frist, Zuständigkeit, Beweise, Kosten)?",
    ],
    "scenario": [
        "Sachverhalt: {scenario} Frage: {question}",
        "Fall: {scenario} {question}",
        "Kontext: {scenario} {question} (Frist/Zuständigkeit/Nachweis)",
        "{scenario} {question} - bitte mit Hinweis zu Fristen und Zuständigkeit.",
    ],
    "procedural": [
        "Wie läuft das Verfahren bei {issue} ab (Frist, Antrag, Nachweise, Kosten)?",
        "{issue}: Welche Fristen gelten, was sind typische Nachweise und wer entscheidet?",
        "Bei {issue}: Welche Rechtsmittel gibt es und hat eine Beschwerde aufschiebende Wirkung?",
        "{issue}: Zuständigkeit und Verfahrensschritte (Antrag/Bescheid/Beschwerde).",
        "Welche Formvorschriften gelten bei {issue} (schriftlich, Frist, Begründung)?",
    ],
    "authority": [
        "Welche Stelle ist zuständig bei {issue} (z.B. {authority})?",
        "Bei {issue}: Muss ich zu {authority} oder zum Gericht?",
        "{issue}: Zuständigkeit {authority} vs Gericht - und welche Frist?",
        "{issue}: Wie bringe ich das bei {authority} ein (Form/Frist)?",
    ],
    "keyword": [
        "{keywords}",
        "{keywords} frist zuständigkeit",
        "{keywords} verfahren beschwerde",
        "{keywords} nachweis kosten",
        "{keywords} bescheid frist",
    ],
    "fragment": [
        "{issue} {city}",
        "{issue} {time}",
        "{issue} {authority}",
        "{issue} {amount_kw}",
        "{issue} {channel} {evidence}",
        "{issue} frist",
        "{issue} zuständig",
        "{issue} beschwerde",
    ],
}
QUESTION_FORMS = [
    "Welche Regelungen sind einschlägig?",
    "Welche Rechte und Pflichten bestehen?",
    "Welche Ansprüche kann ich geltend machen?",
    "Welche Rechtsfolgen drohen bei Verstoß?",
    "Welche Fristen und welches Verfahren sind zu beachten?",
]

def generate_queries_for_topic(
    *,
    topic_id: str,
    issue: str,
    law: str,
    seed: int,
    variants_per_style: int,
    law_mention_prob: float,
    keyword_law_mention_prob: float,
    surface_noise_prob: float,
    law_context_prob: float,
    topic_term_prob: float,
    issue_term_prob: float,
    keyword_term_prob: float,
) -> List[Dict[str, str]]:
    # Context is partly law-conditioned to reduce unrealistic boilerplate.
    ctx = topic_context(topic_id, law, seed, law_context_prob)

    base_rng = random.Random((seed + stable_int(f"{topic_id}:{law}:base")) & 0xFFFFFFFF)

    # Optional per-topic lexicon token (no law abbreviation).
    term_pool = LAW_TERMS.get(law, [])
    topic_term = None
    if term_pool and base_rng.random() < topic_term_prob:
        topic_term = base_rng.choice(term_pool)

    scenario = build_scenario(issue, ctx, base_rng, term=topic_term)
    question = base_rng.choice(QUESTION_FORMS)

    # Keyword source intentionally excludes the law token. Term inclusion is optional.
    k_source = f"{issue} {ctx['city']} {ctx['amount_kw']} {ctx['channel']} {ctx['evidence']} {ctx['time']} {ctx['authority']}"
    if topic_term and base_rng.random() < keyword_term_prob:
        k_source = f"{k_source} {topic_term}"
    keywords = extract_keywords(k_source, max_tokens=11)

    def enrich_issue(rng: random.Random) -> str:
        if not term_pool or rng.random() >= issue_term_prob:
            return issue
        t = topic_term if (topic_term and rng.random() < 0.65) else rng.choice(term_pool)
        return rng.choice([f"{issue} ({t})", f"{issue} - {t}", f"{t}: {issue}"])

    out: List[Dict[str, str]] = []
    for style in STYLES:
        for v in range(1, variants_per_style + 1):
            sseed = (seed + stable_int(f"{topic_id}:{law}:{style}:v{v}")) & 0xFFFFFFFF
            rng = random.Random(sseed)

            template = rng.choice(STYLE_TEMPLATES[style])
            text = template.format(
                issue=enrich_issue(rng),
                scenario=scenario,
                question=question,
                keywords=keywords,
                authority=ctx["authority"],
                city=ctx["city"],
                time=ctx["time"],
                amount_kw=ctx["amount_kw"],
                channel=ctx["channel"],
                evidence=ctx["evidence"],
            )
            text = normalize_ws(text)

            # Optional law hint (kept low)
            p = keyword_law_mention_prob if style == "keyword" else law_mention_prob
            if rng.random() < p:
                text = inject_law_hint(text, law, rng)

            text = maybe_apply_surface_noise(text, rng, surface_noise_prob)

            out.append(
                {
                    "query_id": f"{topic_id}_{style}_v{v:02d}",
                    "topic_id": topic_id,
                    "query_text": text,
                    "consensus_law": law,
                    "style": style,
                    "issue": issue,
                }
            )
    return out


# -------------------------
# Allocation and sampling
# -------------------------

def target_counts(total: int, labels: Sequence[str]) -> Dict[str, int]:
    n_labels = len(labels)
    base = total // n_labels
    rem = total - base * n_labels
    labs = sorted(labels)
    out = {lab: base for lab in labs}
    for i in range(rem):
        out[labs[i]] += 1
    return out

def sample_stratified(
    pool: List[Dict[str, str]],
    target: Dict[str, int],
    seed: int,
    forbid_texts: Set[str],
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    by_law: Dict[str, List[Dict[str, str]]] = {l: [] for l in target}
    for r in pool:
        lab = r["consensus_law"]
        if lab in by_law:
            by_law[lab].append(r)

    for law in sorted(by_law):
        rng.shuffle(by_law[law])

    selected: List[Dict[str, str]] = []
    used_texts: Set[str] = set(forbid_texts)

    for law in sorted(target):
        need = target[law]
        picks: List[Dict[str, str]] = []
        for r in by_law[law]:
            if len(picks) >= need:
                break
            t = r["query_text"]
            if t in used_texts:
                continue
            used_texts.add(t)
            picks.append(r)
        if len(picks) != need:
            raise RuntimeError(
                f"Not enough unique candidates for {law}: need={need}, got={len(picks)}. "
                f"Increase --variants_per_style or --candidate_oversupply."
            )
        selected.extend(picks)

    rng.shuffle(selected)
    return selected

# --- New: stratification by (law, style) and i.i.d. splitting ---

def target_counts_law_style(total_by_law: Dict[str, int], styles: Sequence[str], seed: int) -> Dict[Tuple[str, str], int]:
    """Expand per-law target counts into (law, style) counts.

    For each law, counts are distributed near-uniformly over styles.
    The remainder is assigned using a deterministic per-law shuffle (seeded) to avoid systematic bias toward early styles.
    """
    styles_list = list(styles)
    n_styles = len(styles_list)
    out: Dict[Tuple[str, str], int] = {}

    for law in sorted(total_by_law):
        total = total_by_law[law]
        base = total // n_styles
        rem = total - base * n_styles

        # Deterministic per-law shuffle to spread remainder across styles.
        rng = random.Random((seed + stable_int(f"{law}:style_rem")) & 0xFFFFFFFF)
        order = styles_list[:]
        rng.shuffle(order)

        for style in styles_list:
            out[(law, style)] = base

        for i in range(rem):
            out[(law, order[i])] += 1

    return out

def sample_stratified_grid(
    pool: List[Dict[str, str]],
    target: Dict[Tuple[str, str], int],
    seed: int,
    forbid_texts: Set[str],
) -> List[Dict[str, str]]:
    """Sample exactly `target[(law, style)]` rows per (law, style) bucket."""
    rng = random.Random(seed)
    keys = sorted(target)
    by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = {k: [] for k in keys}

    for r in pool:
        k = (r["consensus_law"], r["style"])
        if k in by_key:
            by_key[k].append(r)

    for k in keys:
        rng.shuffle(by_key[k])

    selected: List[Dict[str, str]] = []
    used_texts: Set[str] = set(forbid_texts)

    for k in keys:
        need = target[k]
        picks: List[Dict[str, str]] = []
        for r in by_key[k]:
            if len(picks) >= need:
                break
            t = r["query_text"]
            if t in used_texts:
                continue
            used_texts.add(t)
            picks.append(r)

        if len(picks) != need:
            law, style = k
            raise RuntimeError(
                f"Not enough unique candidates for (law={law}, style={style}): need={need}, got={len(picks)}. "
                f"Increase --variants_per_style or --candidate_oversupply, or reduce --surface_noise_prob."
            )
        selected.extend(picks)

    rng.shuffle(selected)
    return selected

def split_train_test_stratified_grid(
    pool: List[Dict[str, str]],
    train_target: Dict[Tuple[str, str], int],
    test_target: Dict[Tuple[str, str], int],
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Partition a single pool into TRAIN and TEST with no text overlap, stratified by (law, style)."""
    rng = random.Random(seed)

    keys = sorted(set(train_target) | set(test_target))
    by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = {k: [] for k in keys}

    for r in pool:
        k = (r["consensus_law"], r["style"])
        if k in by_key:
            by_key[k].append(r)

    for k in keys:
        rng.shuffle(by_key[k])

    used_texts: Set[str] = set()
    train_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for k in keys:
        need_tr = train_target.get(k, 0)
        need_te = test_target.get(k, 0)

        tr_picks: List[Dict[str, str]] = []
        te_picks: List[Dict[str, str]] = []

        for r in by_key[k]:
            if len(tr_picks) < need_tr:
                t = r["query_text"]
                if t in used_texts:
                    continue
                used_texts.add(t)
                tr_picks.append(r)
            elif len(te_picks) < need_te:
                t = r["query_text"]
                if t in used_texts:
                    continue
                used_texts.add(t)
                te_picks.append(r)

            if len(tr_picks) == need_tr and len(te_picks) == need_te:
                break

        if len(tr_picks) != need_tr or len(te_picks) != need_te:
            law, style = k
            raise RuntimeError(
                f"Not enough unique candidates for (law={law}, style={style}): "
                f"need_train={need_tr}, got_train={len(tr_picks)}; "
                f"need_test={need_te}, got_test={len(te_picks)}. "
                f"Increase --variants_per_style or --candidate_oversupply, or reduce --surface_noise_prob."
            )

        train_rows.extend(tr_picks)
        test_rows.extend(te_picks)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows
def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--train_n", type=int, default=30000)
    ap.add_argument("--test_n", type=int, default=5000)
    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument("--variants_per_style", type=int, default=3)

    # Split policy
    ap.add_argument(
        "--split_mode",
        choices=["iid", "topic_disjoint"],
        default="iid",
        help="iid: TRAIN/TEST share the same topic mixture (recommended). topic_disjoint: no topic appears in both splits.",
    )

    # Leakage control
    ap.add_argument("--law_mention_prob", type=float, default=0.12)
    ap.add_argument("--keyword_law_mention_prob", type=float, default=0.25)

    # Noise control
    ap.add_argument("--surface_noise_prob", type=float, default=0.06)

    # Richness controls (improve topical realism and discriminative signal)
    ap.add_argument(
        "--law_context_prob",
        type=float,
        default=0.65,
        help="Probability of using law-specific context overrides (authority/counterparty) when available.",
    )
    ap.add_argument(
        "--topic_term_prob",
        type=float,
        default=0.30,
        help="Probability of selecting a per-topic law-lexicon term (no law abbreviations).",
    )
    ap.add_argument(
        "--issue_term_prob",
        type=float,
        default=0.35,
        help="Per-query probability of enriching the issue with a law-lexicon term.",
    )
    ap.add_argument(
        "--keyword_term_prob",
        type=float,
        default=0.35,
        help="Probability of including the per-topic term into keyword-style queries.",
    )

    # Candidate oversupply safety factor (per law, per split or combined depending on split_mode)
    ap.add_argument("--candidate_oversupply", type=float, default=2.0)

    args = ap.parse_args()
    seed = args.seed
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Sanity: ensure no duplicate labels
    if len(set(LAWS)) != len(LAWS):
        raise RuntimeError("LAWS contains duplicates; fix the label universe.")
    if len(LAWS) < 2:
        raise RuntimeError("LAWS is unexpectedly small.")

    train_target_law = target_counts(args.train_n, LAWS)
    test_target_law = target_counts(args.test_n, LAWS)

    # Also match style marginals (near-uniform within each law).
    train_target = target_counts_law_style(train_target_law, STYLES, seed + 777)
    test_target = target_counts_law_style(test_target_law, STYLES, seed + 777)

    queries_per_topic = len(STYLES) * args.variants_per_style

    base_specs = base_law_specs()
    topics_by_law: Dict[str, List[Tuple[str, str]]] = {}

    # --- Topic allocation ---
    extra_topics_per_law = 8  # slack for allocation randomness

    if args.split_mode == "topic_disjoint":
        # Old behavior: allocate disjoint topic sets for TRAIN and TEST.
        train_topics_per_law: Dict[str, int] = {}
        test_topics_per_law: Dict[str, int] = {}
        for law in sorted(LAWS):
            train_topics_per_law[law] = max(
                6, ceil((train_target_law[law] * args.candidate_oversupply) / max(1, queries_per_topic))
            )
            test_topics_per_law[law] = max(
                3, ceil((test_target_law[law] * args.candidate_oversupply) / max(1, queries_per_topic))
            )

        for law in sorted(LAWS):
            if law not in base_specs:
                raise RuntimeError(f"Missing base spec for {law}")

            spec = expand_spec_with_facets(law, base_specs[law], seed=seed)
            need_topics = train_topics_per_law[law] + test_topics_per_law[law] + extra_topics_per_law
            issues = generate_issues_for_law(law, spec, min_count=need_topics, seed=seed)
            topics_by_law[law] = [(f"{law}_T{i:03d}", issues[i - 1]) for i in range(1, need_topics + 1)]

        rng = random.Random(seed)
        train_topics: List[Tuple[str, str, str]] = []  # topic_id, issue, law
        test_topics: List[Tuple[str, str, str]] = []
        for law in sorted(LAWS):
            topics = topics_by_law[law][:]
            rng.shuffle(topics)
            tr = topics[: train_topics_per_law[law]]
            te = topics[
                train_topics_per_law[law] : train_topics_per_law[law] + test_topics_per_law[law]
            ]
            train_topics.extend([(tid, issue, law) for tid, issue in tr])
            test_topics.extend([(tid, issue, law) for tid, issue in te])

        def build_pool(topics: List[Tuple[str, str, str]], split_seed: int) -> List[Dict[str, str]]:
            pool: List[Dict[str, str]] = []
            for tid, issue, law in topics:
                pool.extend(
                    generate_queries_for_topic(
                        topic_id=tid,
                        issue=issue,
                        law=law,
                        seed=split_seed,
                        variants_per_style=args.variants_per_style,
                        law_mention_prob=args.law_mention_prob,
                        keyword_law_mention_prob=args.keyword_law_mention_prob,
                        surface_noise_prob=args.surface_noise_prob,
                        law_context_prob=args.law_context_prob,
                        topic_term_prob=args.topic_term_prob,
                        issue_term_prob=args.issue_term_prob,
                        keyword_term_prob=args.keyword_term_prob,
                    )
                )
            return pool

        train_pool = build_pool(train_topics, seed + 101)
        test_pool = build_pool(test_topics, seed + 202)

        train_rows = sample_stratified_grid(train_pool, train_target, seed + 303, forbid_texts=set())
        forbid = {r["query_text"] for r in train_rows}
        test_rows = sample_stratified_grid(test_pool, test_target, seed + 404, forbid_texts=forbid)

        split_meta = {
            "train_topics_per_law": train_topics_per_law,
            "test_topics_per_law": test_topics_per_law,
            "extra_topics_per_law": extra_topics_per_law,
        }

    else:
        # New default: single shared topic pool; TRAIN/TEST are stratified partitions of that pool.
        topics_per_law: Dict[str, int] = {}
        for law in sorted(LAWS):
            total_need = train_target_law[law] + test_target_law[law]
            topics_per_law[law] = max(
                8, ceil((total_need * args.candidate_oversupply) / max(1, queries_per_topic))
            )

        for law in sorted(LAWS):
            if law not in base_specs:
                raise RuntimeError(f"Missing base spec for {law}")

            spec = expand_spec_with_facets(law, base_specs[law], seed=seed)
            need_topics = topics_per_law[law] + extra_topics_per_law
            issues = generate_issues_for_law(law, spec, min_count=need_topics, seed=seed)
            topics_by_law[law] = [(f"{law}_T{i:03d}", issues[i - 1]) for i in range(1, need_topics + 1)]

        all_topics: List[Tuple[str, str, str]] = []
        for law in sorted(LAWS):
            for tid, issue in topics_by_law[law]:
                all_topics.append((tid, issue, law))

        def build_pool(topics: List[Tuple[str, str, str]], split_seed: int) -> List[Dict[str, str]]:
            pool: List[Dict[str, str]] = []
            for tid, issue, law in topics:
                pool.extend(
                    generate_queries_for_topic(
                        topic_id=tid,
                        issue=issue,
                        law=law,
                        seed=split_seed,
                        variants_per_style=args.variants_per_style,
                        law_mention_prob=args.law_mention_prob,
                        keyword_law_mention_prob=args.keyword_law_mention_prob,
                        surface_noise_prob=args.surface_noise_prob,
                        law_context_prob=args.law_context_prob,
                        topic_term_prob=args.topic_term_prob,
                        issue_term_prob=args.issue_term_prob,
                        keyword_term_prob=args.keyword_term_prob,
                    )
                )
            return pool

        # One pool only -> no query_id collisions even when topics appear in both splits.
        pool = build_pool(all_topics, seed + 111)

        train_rows, test_rows = split_train_test_stratified_grid(pool, train_target, test_target, seed + 303)

        split_meta = {
            "topics_per_law": topics_per_law,
            "extra_topics_per_law": extra_topics_per_law,
        }

    # --- Write outputs ---
    train_path = outdir / "train.jsonl"
    test_path = outdir / "test.jsonl"
    meta_path = outdir / "meta.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    def _count(rows: List[Dict[str, str]], fields: Tuple[str, ...]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rows:
            k = "||".join(r[f] for f in fields)
            out[k] = out.get(k, 0) + 1
        return out

    meta = {
        "seed": seed,
        "split_mode": args.split_mode,
        "train_n": args.train_n,
        "test_n": args.test_n,
        "laws": sorted(LAWS),
        "n_laws": len(LAWS),
        "styles": STYLES,
        "variants_per_style": args.variants_per_style,
        "queries_per_topic": queries_per_topic,
        "law_mention_prob": args.law_mention_prob,
        "keyword_law_mention_prob": args.keyword_law_mention_prob,
        "surface_noise_prob": args.surface_noise_prob,
        "law_context_prob": args.law_context_prob,
        "topic_term_prob": args.topic_term_prob,
        "issue_term_prob": args.issue_term_prob,
        "keyword_term_prob": args.keyword_term_prob,
        "candidate_oversupply": args.candidate_oversupply,
        "train_target_counts_by_law": train_target_law,
        "test_target_counts_by_law": test_target_law,
        "train_target_counts_by_law_style": {f"{k[0]}||{k[1]}": v for k, v in train_target.items()},
        "test_target_counts_by_law_style": {f"{k[0]}||{k[1]}": v for k, v in test_target.items()},
        "realized_train_counts_by_law_style": _count(train_rows, ("consensus_law", "style")),
        "realized_test_counts_by_law_style": _count(test_rows, ("consensus_law", "style")),
        "files": {"train": str(train_path), "test": str(test_path), "meta": str(meta_path)},
    }
    meta.update(split_meta)

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(test_rows)} test rows to {test_path}")
    print(f"Wrote meta to {meta_path}")

if __name__ == "__main__":
    main()
