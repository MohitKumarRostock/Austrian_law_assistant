#!/usr/bin/env python3
"""query_set.py

Enriched query set for Austrian law retrieval evaluation through:
  - scenario phrasing (without explicit label)
  - legal-register phrasing
  - reframing of interrogatives (Wann/In welchen Fällen/…)
  - conservative Austrian legal synonyms (when applicable)
  - keyword-style and short-form queries

It preserves the ORIGINAL law label universe (no new consensus_law codes).

Exports:
  - BASE_QUERY_SET (n=100)
  - QUERY_SET (n=800) -> 100 * 8 variants
  - TRAIN_QUERY_SET (n=600) -> first 6 variants per base query
  - TEST_QUERY_SET (n=200) -> last 2 variants per base query
"""

from __future__ import annotations
from typing import Dict, List

BASE_QUERY_SET: List[Dict[str, str]] = [
    {
        'query_id': 'Q1_arbeitgeber_haftung',
        'query_text': 'Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen',
        'query_text': 'Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv',
        'query_text': 'In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren',
        'query_text': 'Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt',
        'query_text': 'Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche',
        'query_text': 'Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16',
        'query_text': 'Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung',
        'query_text': 'Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber',
        'query_text': 'Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch',
        'query_text': 'Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q11_mobbing_arbeit',
        'query_text': 'Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung',
        'query_text': 'Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte',
        'query_text': 'Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden',
        'query_text': 'Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q15_berufung_strafurteil',
        'query_text': 'Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber',
        'query_text': 'Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung',
        'query_text': 'Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q18_betrug_strafbar',
        'query_text': 'Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen',
        'query_text': 'Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall',
        'query_text': 'Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch',
        'query_text': 'Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q22_hund_haftung',
        'query_text': 'Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument',
        'query_text': 'Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft',
        'query_text': 'Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung',
        'query_text': 'Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag',
        'query_text': 'Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum',
        'query_text': 'Kann ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz',
        'query_text': 'Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz',
        'query_text': 'Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten',
        'query_text': 'Darf ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern',
        'query_text': 'In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit',
        'query_text': 'Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung',
        'query_text': 'Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden',
        'query_text': 'Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl',
        'query_text': 'Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv',
        'query_text': 'Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung',
        'query_text': 'Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten',
        'query_text': 'Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch',
        'query_text': 'Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift',
        'query_text': 'Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen',
        'query_text': 'Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot',
        'query_text': 'Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte',
        'query_text': 'Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen',
        'query_text': 'Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung',
        'query_text': 'Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter',
        'query_text': 'Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite',
        'query_text': 'Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene',
        'query_text': 'Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website',
        'query_text': 'Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten',
        'query_text': 'Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar',
        'query_text': 'Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei',
        'query_text': 'Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer',
        'query_text': 'Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren',
        'query_text': 'Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen',
        'query_text': 'Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer',
        'query_text': 'Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung',
        'query_text': 'Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer',
        'query_text': 'Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid',
        'query_text': 'Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen',
        'query_text': 'Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern',
        'query_text': 'Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft',
        'query_text': 'Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung',
        'query_text': 'Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware',
        'query_text': 'Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung',
        'query_text': 'Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten',
        'query_text': 'Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage',
        'query_text': 'Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt',
        'query_text': 'Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen',
        'query_text': 'Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung',
        'query_text': 'Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung',
        'query_text': 'Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang',
        'query_text': 'Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo',
        'query_text': 'Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung',
        'query_text': 'Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q75_impressumspflicht_website',
        'query_text': 'Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung',
        'query_text': 'Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf',
        'query_text': 'Wie läuft die rechtliche Gründung einer GmbH in Österreich ab?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse',
        'query_text': 'Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft',
        'query_text': 'Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh',
        'query_text': 'Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren',
        'query_text': 'Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte',
        'query_text': 'Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht',
        'query_text': 'Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid',
        'query_text': 'Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren',
        'query_text': 'Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit',
        'query_text': 'Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft',
        'query_text': 'Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job',
        'query_text': 'Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz',
        'query_text': 'Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15',
        'query_text': 'Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen',
        'query_text': 'Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung',
        'query_text': 'Darf ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer',
        'query_text': 'Wie lange kann ich Kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle',
        'query_text': 'Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle',
        'query_text': 'An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung',
        'query_text': 'Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt',
        'query_text': 'Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken',
        'query_text': 'Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung',
        'query_text': 'Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr',
        'query_text': 'Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
]
QUERY_SET: List[Dict[str, str]] = [
    {
        'query_id': 'Q1_arbeitgeber_haftung',
        'query_text': 'Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v3',
        'query_text': 'Angenommen, Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v4',
        'query_text': 'Wann haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v6',
        'query_text': 'Unter welchen Voraussetzungen haftet der Dienstgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v7',
        'query_text': 'welchen voraussetzungen haftet arbeitgeber schäden arbeitnehmers gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v8',
        'query_text': 'welchen voraussetzungen haftet arbeitgeber schäden arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen',
        'query_text': 'Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v3',
        'query_text': 'Angenommen, Welche Rechte hat der Betriebsrat bei Kündigungen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v6',
        'query_text': 'rechte betriebsrat kündigungen gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v7',
        'query_text': 'rechte betriebsrat kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v8',
        'query_text': 'rechte betriebsrat kündigungen Bitte mit Verweisen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv',
        'query_text': 'In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v3',
        'query_text': 'Angenommen, In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v4',
        'query_text': 'Wie ist die Rechtslage bei In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v6',
        'query_text': 'welchen fällen haftet dienstgeber gegenüber trägern sozialversicherung gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v7',
        'query_text': 'welchen fällen haftet dienstgeber gegenüber trägern sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v8',
        'query_text': 'welchen fällen haftet dienstgeber gegenüber trägern sozialversicherung Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren',
        'query_text': 'Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Beschuldigter in einem Strafverfahren. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v6',
        'query_text': 'rechte habe beschuldigter strafverfahren gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v7',
        'query_text': 'rechte habe beschuldigter strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v8',
        'query_text': 'rechte habe beschuldigter strafverfahren Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt',
        'query_text': 'Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v3',
        'query_text': 'Angenommen, Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v4',
        'query_text': 'Wie ist kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt gesetzlich geregelt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v6',
        'query_text': 'jemanden klagen geld schuldet zahlt gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v7',
        'query_text': 'jemanden klagen geld schuldet zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v8',
        'query_text': 'jemanden klagen geld schuldet zahlt Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche',
        'query_text': 'Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v3',
        'query_text': 'Angenommen, Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v4',
        'query_text': 'Ist es zulässig, dass der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v6',
        'query_text': 'Darf der Dienstgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v7',
        'query_text': 'arbeitgeber kündigen meine ansprüche geltend mache gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v8',
        'query_text': 'arbeitgeber kündigen meine ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16',
        'query_text': 'Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v3',
        'query_text': 'Angenommen, Was darf ich tun, wenn ich 16 bin. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v4',
        'query_text': 'Wie ist die Rechtslage bei Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v6',
        'query_text': 'tun bin gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v7',
        'query_text': 'tun bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v8',
        'query_text': 'tun bin Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung',
        'query_text': 'Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v3',
        'query_text': 'Angenommen, Wann darf der Vermieter den Mietvertrag kündigen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v6',
        'query_text': 'Wann darf der Bestandgeber den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v7',
        'query_text': 'vermieter mietvertrag kündigen gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v8',
        'query_text': 'vermieter mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber',
        'query_text': 'Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v6',
        'query_text': 'Welche Rechte habe ich nach dem DSG, wenn mein Dienstgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v7',
        'query_text': 'rechte habe dsg mein arbeitgeber meine daten speichert gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v8',
        'query_text': 'rechte habe dsg mein arbeitgeber meine daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch',
        'query_text': 'Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v3',
        'query_text': 'Angenommen, Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v6',
        'query_text': 'habe anspruch sozialhilfe mindestsicherung gesetzliche Bestimmungen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v7',
        'query_text': 'habe anspruch sozialhilfe mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v8',
        'query_text': 'habe anspruch sozialhilfe mindestsicherung Bitte mit Verweisen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q11_mobbing_arbeit',
        'query_text': 'Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v3',
        'query_text': 'Angenommen, Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v6',
        'query_text': 'liegt mobbing arbeitsplatz dagegen tun gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v7',
        'query_text': 'liegt mobbing arbeitsplatz dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v8',
        'query_text': 'liegt mobbing arbeitsplatz dagegen tun Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung',
        'query_text': 'Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v3',
        'query_text': 'Angenommen, Ist eine Kündigung wegen Schwangerschaft zulässig. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v4',
        'query_text': 'Wie ist die Rechtslage bei Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v6',
        'query_text': 'Ist eine Auflösung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v7',
        'query_text': 'kündigung wegen schwangerschaft zulässig gesetzliche Bestimmungen?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v8',
        'query_text': 'kündigung wegen schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte',
        'query_text': 'Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Mieter bei einer Mieterhöhung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v6',
        'query_text': 'Welche Rechte habe ich als Bestandnehmer bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v7',
        'query_text': 'rechte habe mieter mieterhöhung gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v8',
        'query_text': 'rechte habe mieter mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden',
        'query_text': 'Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v3',
        'query_text': 'Angenommen, Muss ich dem Vermieter jeden Besucher melden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v4',
        'query_text': 'Wie ist die Rechtslage bei Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v6',
        'query_text': 'Muss ich dem Bestandgeber jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v7',
        'query_text': 'vermieter jeden besucher melden gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v8',
        'query_text': 'vermieter jeden besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q15_berufung_strafurteil',
        'query_text': 'Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für eine Berufung gegen ein Strafurteil. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v6',
        'query_text': 'fristen gelten berufung strafurteil gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v7',
        'query_text': 'fristen gelten berufung strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v8',
        'query_text': 'fristen gelten berufung strafurteil Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber',
        'query_text': 'Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v3',
        'query_text': 'Angenommen, Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v4',
        'query_text': 'Ist es zulässig, dass mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v6',
        'query_text': 'Darf mein Dienstgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v7',
        'query_text': 'mein arbeitgeber meine dienstlichen e-mails zustimmung lesen gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v8',
        'query_text': 'mein arbeitgeber meine dienstlichen e-mails meine zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung',
        'query_text': 'Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v3',
        'query_text': 'Angenommen, Welche Daten darf eine Behörde über mich speichern. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v6',
        'query_text': 'Welche Daten darf eine Verwaltungsbehörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v7',
        'query_text': 'daten behörde speichern gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v8',
        'query_text': 'daten behörde speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q18_betrug_strafbar',
        'query_text': 'Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v3',
        'query_text': 'Angenommen, Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v6',
        'query_text': 'mache wegen betrugs strafbar jemanden täusche gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v7',
        'query_text': 'mache wegen betrugs strafbar jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v8',
        'query_text': 'mache wegen betrugs strafbar jemanden täusche Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen',
        'query_text': 'Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v3',
        'query_text': 'Angenommen, Was passiert, wenn ich keine Steuern zahle. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v4',
        'query_text': 'Wie ist die Rechtslage bei Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v6',
        'query_text': 'passiert steuern zahle gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v7',
        'query_text': 'passiert steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v8',
        'query_text': 'passiert steuern zahle Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall',
        'query_text': 'Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v3',
        'query_text': 'Angenommen, Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v6',
        'query_text': 'ansprüche habe verkehrsunfall schmerzensgeld gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v7',
        'query_text': 'ansprüche habe verkehrsunfall schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v8',
        'query_text': 'ansprüche habe verkehrsunfall schmerzensgeld Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch',
        'query_text': 'Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v3',
        'query_text': 'Angenommen, Habe ich Anspruch auf Elternkarenz und wie lange. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v4',
        'query_text': 'Wie ist die Rechtslage bei Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v6',
        'query_text': 'habe anspruch elternkarenz lange gesetzliche Bestimmungen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v7',
        'query_text': 'habe anspruch elternkarenz lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v8',
        'query_text': 'habe anspruch elternkarenz lange Bitte mit Verweisen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q22_hund_haftung',
        'query_text': 'Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v3',
        'query_text': 'Angenommen, Wer haftet, wenn mein Hund jemanden verletzt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v6',
        'query_text': 'haftet mein hund jemanden verletzt gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v7',
        'query_text': 'haftet mein hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v8',
        'query_text': 'haftet mein hund jemanden verletzt Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument',
        'query_text': 'Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Konsument bei einem Online-Kauf. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v6',
        'query_text': 'rechte habe konsument online-kauf gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v7',
        'query_text': 'rechte habe konsument online-kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v8',
        'query_text': 'rechte habe konsument online-kauf Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft',
        'query_text': 'Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v3',
        'query_text': 'Angenommen, Wann ist eine Videoüberwachung in einem Geschäft zulässig. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v6',
        'query_text': 'videoüberwachung geschäft zulässig gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v7',
        'query_text': 'videoüberwachung geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v8',
        'query_text': 'videoüberwachung geschäft zulässig Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung',
        'query_text': 'Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v3',
        'query_text': 'Angenommen, Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v4',
        'query_text': 'Ist es zulässig, dass der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v6',
        'query_text': 'Darf der Bestandgeber die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v7',
        'query_text': 'vermieter kaution einbehalten nur normale abnutzung vorliegt gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v8',
        'query_text': 'vermieter kaution einbehalten nur normale abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag',
        'query_text': 'Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v3',
        'query_text': 'Angenommen, Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v4',
        'query_text': 'Wie ist die Rechtslage bei Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v6',
        'query_text': 'mündlicher vertrag rechtsverbindlich gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v7',
        'query_text': 'mündlicher vertrag rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v8',
        'query_text': 'mündlicher vertrag rechtsverbindlich Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum',
        'query_text': 'Kann ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v3',
        'query_text': 'Angenommen, Kann ich einen Kaufvertrag wegen Irrtums anfechten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v4',
        'query_text': 'Ist es möglich, dass ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v6',
        'query_text': 'kaufvertrag wegen irrtums anfechten gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v7',
        'query_text': 'kaufvertrag wegen irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v8',
        'query_text': 'kaufvertrag wegen irrtums anfechten Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz',
        'query_text': 'Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v3',
        'query_text': 'Angenommen, Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v6',
        'query_text': 'haftet kind freund etwas kaputt macht gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v7',
        'query_text': 'haftet kind freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v8',
        'query_text': 'haftet kind freund etwas kaputt macht Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz',
        'query_text': 'Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v3',
        'query_text': 'Angenommen, Wann verjähren Schadenersatzansprüche nach österreichischem Recht. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v6',
        'query_text': 'verjähren schadenersatzansprüche recht gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v7',
        'query_text': 'verjähren schadenersatzansprüche recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v8',
        'query_text': 'verjähren schadenersatzansprüche recht Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten',
        'query_text': 'Darf ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v3',
        'query_text': 'Angenommen, Darf ich ein gefundenes Handy einfach behalten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v4',
        'query_text': 'Ist es zulässig, dass ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v6',
        'query_text': 'gefundenes handy einfach behalten gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v7',
        'query_text': 'gefundenes handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v8',
        'query_text': 'gefundenes handy einfach behalten Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern',
        'query_text': 'In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v3',
        'query_text': 'Angenommen, In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v4',
        'query_text': 'Wie ist die Rechtslage bei In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v6',
        'query_text': 'welchen fällen habe anspruch unterhalt meinen eltern gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v7',
        'query_text': 'welchen fällen habe anspruch unterhalt meinen eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v8',
        'query_text': 'welchen fällen habe anspruch unterhalt meinen eltern Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit',
        'query_text': 'Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v3',
        'query_text': 'Angenommen, Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v6',
        'query_text': 'mitbestimmungsrechte betriebsrat änderungen arbeitszeit gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v7',
        'query_text': 'mitbestimmungsrechte betriebsrat änderungen arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v8',
        'query_text': 'mitbestimmungsrechte betriebsrat änderungen arbeitszeit Bitte mit Verweisen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung',
        'query_text': 'Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v3',
        'query_text': 'Angenommen, Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v4',
        'query_text': 'Wie ist wird ein Betriebsrat in einem Unternehmen gegründet und gewählt gesetzlich geregelt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v6',
        'query_text': 'betriebsrat unternehmen gegründet gewählt gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v7',
        'query_text': 'betriebsrat unternehmen gegründet gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v8',
        'query_text': 'betriebsrat unternehmen gegründet gewählt Bitte mit Verweisen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden',
        'query_text': 'Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v3',
        'query_text': 'Angenommen, Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v4',
        'query_text': 'Ist es zulässig, dass der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v6',
        'query_text': 'Darf der Dienstgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v7',
        'query_text': 'arbeitgeber zustimmung betriebsrats dauerhaft überstunden anordnen gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v8',
        'query_text': 'arbeitgeber zustimmung betriebsrats dauerhaft überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl',
        'query_text': 'Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v6',
        'query_text': 'Welche Rechte habe ich als Dienstnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v7',
        'query_text': 'rechte habe arbeitnehmer betriebsratswahl gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v8',
        'query_text': 'rechte habe arbeitnehmer betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv',
        'query_text': 'Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v3',
        'query_text': 'Angenommen, Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v6',
        'query_text': 'pflichten habe dienstnehmer gegenüber sozialversicherung gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v7',
        'query_text': 'pflichten habe dienstnehmer gegenüber sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v8',
        'query_text': 'pflichten habe dienstnehmer gegenüber sozialversicherung Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung',
        'query_text': 'Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v3',
        'query_text': 'Angenommen, Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v4',
        'query_text': 'Wie ist die Rechtslage bei Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v6',
        'query_text': 'bin geringfügig beschäftigter krankenversicherung abgesichert gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v7',
        'query_text': 'bin geringfügig beschäftigter krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v8',
        'query_text': 'bin geringfügig beschäftigter krankenversicherung abgesichert Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten',
        'query_text': 'Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v3',
        'query_text': 'Angenommen, Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v6',
        'query_text': 'zahlt meine behandlungskosten arbeitsunfall gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v7',
        'query_text': 'zahlt meine behandlungskosten arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v8',
        'query_text': 'zahlt meine behandlungskosten arbeitsunfall Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch',
        'query_text': 'Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v3',
        'query_text': 'Angenommen, Wann habe ich Anspruch auf eine Invaliditätspension in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v6',
        'query_text': 'habe anspruch invaliditätspension gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v7',
        'query_text': 'habe anspruch invaliditätspension?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v8',
        'query_text': 'habe anspruch invaliditätspension Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift',
        'query_text': 'Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v3',
        'query_text': 'Angenommen, Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v4',
        'query_text': 'Ist es möglich, dass der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v6',
        'query_text': 'Kann der Bestandgeber die Mietzins erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v7',
        'query_text': 'vermieter miete erhöhen neuer lift eingebaut wurde gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v8',
        'query_text': 'vermieter miete erhöhen neuer lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen',
        'query_text': 'Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v6',
        'query_text': 'Welche Fristen gelten für die Anfechtung des Mietzinses beim Gerichtshof oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v7',
        'query_text': 'fristen gelten anfechtung mietzinses beim gericht schlichtungsstelle gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v8',
        'query_text': 'fristen gelten anfechtung mietzinses beim gericht schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot',
        'query_text': 'Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v3',
        'query_text': 'Angenommen, Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v4',
        'query_text': 'Ist es zulässig, dass der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v6',
        'query_text': 'Darf der Bestandgeber mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v7',
        'query_text': 'vermieter haustiere mietvertrag generell verbieten gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v8',
        'query_text': 'vermieter haustiere mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte',
        'query_text': 'Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v6',
        'query_text': 'rechte habe wohnungseigentümer eigentümergemeinschaft gesetzliche Bestimmungen?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v7',
        'query_text': 'rechte habe wohnungseigentümer eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v8',
        'query_text': 'rechte habe wohnungseigentümer eigentümergemeinschaft Bitte mit Verweisen?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen',
        'query_text': 'Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v3',
        'query_text': 'Angenommen, Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v4',
        'query_text': 'Wie ist kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen gesetzlich geregelt?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v6',
        'query_text': 'unternehmen auskunft gespeicherten daten verlangen gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v7',
        'query_text': 'unternehmen auskunft gespeicherten daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v8',
        'query_text': 'unternehmen auskunft gespeicherten daten verlangen Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung',
        'query_text': 'Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v3',
        'query_text': 'Angenommen, Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v4',
        'query_text': 'Wie ist die Rechtslage bei Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v6',
        'query_text': 'welchen bedingungen online-shop meine daten werbezwecken verwenden gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v7',
        'query_text': 'welchen bedingungen online-shop meine daten werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v8',
        'query_text': 'welchen bedingungen online-shop meine daten werbezwecken verwenden Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter',
        'query_text': 'Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v3',
        'query_text': 'Angenommen, Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v4',
        'query_text': 'Ist es zulässig, dass mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v6',
        'query_text': 'mein internetanbieter meine verbindungsdaten speichern auswerten gesetzliche Bestimmungen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v7',
        'query_text': 'mein internetanbieter meine verbindungsdaten speichern auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v8',
        'query_text': 'mein internetanbieter meine verbindungsdaten speichern auswerten Bitte mit Verweisen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite',
        'query_text': 'Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v3',
        'query_text': 'Angenommen, Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v6',
        'query_text': 'regeln gelten cookies tracking webseiten gesetzliche Bestimmungen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v7',
        'query_text': 'regeln gelten cookies tracking webseiten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v8',
        'query_text': 'regeln gelten cookies tracking webseiten Bitte mit Verweisen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene',
        'query_text': 'Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v6',
        'query_text': 'rechte habe meine personenbezogenen daten datenleck veröffentlicht wurden gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v7',
        'query_text': 'rechte habe meine personenbezogenen daten datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v8',
        'query_text': 'rechte habe meine personenbezogenen daten datenleck veröffentlicht wurden Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website',
        'query_text': 'Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v3',
        'query_text': 'Angenommen, Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v4',
        'query_text': 'Ist es zulässig, dass eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v6',
        'query_text': 'schule fotos meine zustimmung ihrer website veröffentlichen gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v7',
        'query_text': 'schule fotos meine zustimmung ihrer website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v8',
        'query_text': 'schule fotos meine zustimmung ihrer website veröffentlichen Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten',
        'query_text': 'Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v3',
        'query_text': 'Angenommen, Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v6',
        'query_text': 'strafrechtlichen folgen drohen fahrschein fahre kontrolle falsche daten angebe gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v7',
        'query_text': 'strafrechtlichen folgen drohen fahrschein fahre kontrolle falsche daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v8',
        'query_text': 'strafrechtlichen folgen drohen fahrschein fahre kontrolle falsche daten angebe Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar',
        'query_text': 'Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v3',
        'query_text': 'Angenommen, Ab wann mache ich mich wegen Körperverletzung strafbar. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v4',
        'query_text': 'Wie ist die Rechtslage bei Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v6',
        'query_text': 'mache wegen körperverletzung strafbar gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v7',
        'query_text': 'mache wegen körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v8',
        'query_text': 'mache wegen körperverletzung strafbar Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei',
        'query_text': 'Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v3',
        'query_text': 'Angenommen, Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v4',
        'query_text': 'Ist es zulässig, dass die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v6',
        'query_text': 'Darf die Polizeibehörde mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v7',
        'query_text': 'polizei mein handy richterlichen beschluss durchsuchen gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v8',
        'query_text': 'polizei mein handy richterlichen beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer',
        'query_text': 'Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v3',
        'query_text': 'Angenommen, Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v4',
        'query_text': 'Wie ist lange darf ich in Österreich in Untersuchungshaft gehalten werden gesetzlich geregelt?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v6',
        'query_text': 'lange untersuchungshaft gehalten gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v7',
        'query_text': 'lange untersuchungshaft gehalten?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v8',
        'query_text': 'lange untersuchungshaft gehalten Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren',
        'query_text': 'Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Opfer in einem Strafverfahren. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v6',
        'query_text': 'rechte habe opfer strafverfahren gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v7',
        'query_text': 'rechte habe opfer strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v8',
        'query_text': 'rechte habe opfer strafverfahren Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen',
        'query_text': 'Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v3',
        'query_text': 'Angenommen, Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v4',
        'query_text': 'Wie ist die Rechtslage bei Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v6',
        'query_text': 'passiert verwaltungsstrafe bezahle gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v7',
        'query_text': 'passiert verwaltungsstrafe bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v8',
        'query_text': 'passiert verwaltungsstrafe bezahle Bitte mit Verweisen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer',
        'query_text': 'Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v3',
        'query_text': 'Angenommen, Welche Strafen drohen bei Trunkenheit am Steuer in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v6',
        'query_text': 'strafen drohen trunkenheit steuer gesetzliche Bestimmungen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v7',
        'query_text': 'strafen drohen trunkenheit steuer?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v8',
        'query_text': 'strafen drohen trunkenheit steuer Bitte mit Verweisen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung',
        'query_text': 'Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v3',
        'query_text': 'Angenommen, Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v6',
        'query_text': 'verjährt verwaltungsübertretung recht gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v7',
        'query_text': 'verjährt verwaltungsübertretung recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v8',
        'query_text': 'verjährt verwaltungsübertretung recht Bitte mit Verweisen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer',
        'query_text': 'Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v6',
        'query_text': 'fristen gelten abgabe einkommensteuererklärung beim finanzamt gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v7',
        'query_text': 'fristen gelten abgabe einkommensteuererklärung beim finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v8',
        'query_text': 'fristen gelten abgabe einkommensteuererklärung beim finanzamt Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid',
        'query_text': 'Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v3',
        'query_text': 'Angenommen, Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v4',
        'query_text': 'Wie ist kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen gesetzlich geregelt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v6',
        'query_text': 'einkommensteuerbescheid beschwerde einlegen gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v7',
        'query_text': 'einkommensteuerbescheid beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v8',
        'query_text': 'einkommensteuerbescheid beschwerde einlegen Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen',
        'query_text': 'Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v3',
        'query_text': 'Angenommen, Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v4',
        'query_text': 'Wie ist die Rechtslage bei Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v6',
        'query_text': 'passiert umsätze meiner steuererklärung verschweige gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v7',
        'query_text': 'passiert umsätze meiner steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v8',
        'query_text': 'passiert umsätze meiner steuererklärung verschweige Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern',
        'query_text': 'Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v3',
        'query_text': 'Angenommen, Muss ich in Österreich auch Auslandseinkünfte versteuern. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v4',
        'query_text': 'Wie ist die Rechtslage bei Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v6',
        'query_text': 'auslandseinkünfte versteuern gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v7',
        'query_text': 'auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v8',
        'query_text': 'auslandseinkünfte versteuern Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft',
        'query_text': 'Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v3',
        'query_text': 'Angenommen, Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v4',
        'query_text': 'Ist es möglich, dass ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v6',
        'query_text': 'geschäft gekauften artikel angabe gründen zurückgeben gesetzliche Bestimmungen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v7',
        'query_text': 'geschäft gekauften artikel angabe gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v8',
        'query_text': 'geschäft gekauften artikel angabe gründen zurückgeben Bitte mit Verweisen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung',
        'query_text': 'Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v3',
        'query_text': 'Angenommen, Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v6',
        'query_text': 'widerrufsfrist gilt online-bestellungen gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v7',
        'query_text': 'widerrufsfrist gilt online-bestellungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v8',
        'query_text': 'widerrufsfrist gilt online-bestellungen Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware',
        'query_text': 'Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v3',
        'query_text': 'Angenommen, Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v4',
        'query_text': 'Ist es zulässig, dass ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v6',
        'query_text': 'unternehmer gewährleistung gebrauchten ausschließen gesetzliche Bestimmungen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v7',
        'query_text': 'unternehmer gewährleistung gebrauchten ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v8',
        'query_text': 'unternehmer gewährleistung gebrauchten ausschließen Bitte mit Verweisen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung',
        'query_text': 'Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v3',
        'query_text': 'Angenommen, Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v4',
        'query_text': 'Wie ist die Rechtslage bei Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v6',
        'query_text': 'meine rechte online-abonnement automatisch verlängert gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v7',
        'query_text': 'meine rechte online-abonnement automatisch verlängert?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v8',
        'query_text': 'meine rechte online-abonnement automatisch verlängert Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten',
        'query_text': 'Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v3',
        'query_text': 'Angenommen, Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v6',
        'query_text': 'informationen unternehmer vertragsabschluss fernabsatz geben gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v7',
        'query_text': 'informationen unternehmer vertragsabschluss fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v8',
        'query_text': 'informationen unternehmer vertragsabschluss fernabsatz geben Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage',
        'query_text': 'Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v3',
        'query_text': 'Angenommen, Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v4',
        'query_text': 'Wie ist die Rechtslage bei Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v6',
        'query_text': 'Vor welchem Gerichtshof muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v7',
        'query_text': 'welchem gericht klagen schuldner anderen bundesland wohnt gesetzliche Bestimmungen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v8',
        'query_text': 'welchem gericht klagen schuldner anderen bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt',
        'query_text': 'Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v3',
        'query_text': 'Angenommen, Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v6',
        'query_text': 'kosten fallen zivilklage trägt erfolgs- misserfolgsfall gesetzliche Bestimmungen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v7',
        'query_text': 'kosten fallen zivilklage trägt erfolgs- misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v8',
        'query_text': 'kosten fallen zivilklage trägt erfolgs- misserfolgsfall Bitte mit Verweisen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen',
        'query_text': 'Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v3',
        'query_text': 'Angenommen, Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v4',
        'query_text': 'Ist es möglich, dass ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v6',
        'query_text': 'verfahrenshilfe beantragen zivilprozess leisten gesetzliche Bestimmungen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v7',
        'query_text': 'verfahrenshilfe beantragen zivilprozess leisten?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v8',
        'query_text': 'verfahrenshilfe beantragen zivilprozess leisten Bitte mit Verweisen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung',
        'query_text': 'Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v3',
        'query_text': 'Angenommen, Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v6',
        'query_text': 'haftet meinem auto fußgänger verletze gesetzliche Bestimmungen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v7',
        'query_text': 'haftet meinem auto fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v8',
        'query_text': 'haftet meinem auto fußgänger verletze Bitte mit Verweisen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung',
        'query_text': 'Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v3',
        'query_text': 'Angenommen, Haftet der Halter oder der Lenker bei einem Autounfall. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v4',
        'query_text': 'Wie ist die Rechtslage bei Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v6',
        'query_text': 'haftet halter lenker autounfall gesetzliche Bestimmungen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v7',
        'query_text': 'haftet halter lenker autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v8',
        'query_text': 'haftet halter lenker autounfall Bitte mit Verweisen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang',
        'query_text': 'Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v3',
        'query_text': 'Angenommen, Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v4',
        'query_text': 'Ist es möglich, dass ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v6',
        'query_text': 'verkehrsunfall verdienstentgang geltend machen gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v7',
        'query_text': 'verkehrsunfall verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v8',
        'query_text': 'verkehrsunfall verdienstentgang geltend machen Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo',
        'query_text': 'Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v3',
        'query_text': 'Angenommen, Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v6',
        'query_text': 'pflichten habe unfall sachschaden straßenverkehrsordnung gesetzliche Bestimmungen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v7',
        'query_text': 'pflichten habe unfall sachschaden straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v8',
        'query_text': 'pflichten habe unfall sachschaden straßenverkehrsordnung Bitte mit Verweisen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung',
        'query_text': 'Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v3',
        'query_text': 'Angenommen, Ab wann muss ich als Unternehmer doppelte Buchhaltung führen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v4',
        'query_text': 'Wie ist die Rechtslage bei Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v6',
        'query_text': 'unternehmer doppelte buchhaltung führen gesetzliche Bestimmungen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v7',
        'query_text': 'unternehmer doppelte buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v8',
        'query_text': 'unternehmer doppelte buchhaltung führen Bitte mit Verweisen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q75_impressumspflicht_website',
        'query_text': 'Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v3',
        'query_text': 'Angenommen, Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v6',
        'query_text': 'angaben impressum unternehmenswebsite enthalten gesetzliche Bestimmungen?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v7',
        'query_text': 'angaben impressum unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v8',
        'query_text': 'angaben impressum unternehmenswebsite enthalten Bitte mit Verweisen?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung',
        'query_text': 'Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v6',
        'query_text': 'rechte habe gesellschafter gmbh einsicht unterlagen gesellschaft gesetzliche Bestimmungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v7',
        'query_text': 'rechte habe gesellschafter gmbh einsicht unterlagen gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v8',
        'query_text': 'rechte habe gesellschafter gmbh einsicht unterlagen gesellschaft Bitte mit Verweisen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf',
        'query_text': 'Wie läuft die rechtliche Gründung einer GmbH in Österreich ab?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie läuft die rechtliche Gründung einer GmbH in Österreich ab?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v3',
        'query_text': 'Angenommen, Wie läuft die rechtliche Gründung einer GmbH in Österreich ab. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v4',
        'query_text': 'Wie ist läuft die rechtliche Gründung einer GmbH in Österreich ab gesetzlich geregelt?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie läuft die rechtliche Gründung einer GmbH in Österreich ab?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v6',
        'query_text': 'läuft rechtliche gründung gmbh gesetzliche Bestimmungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v7',
        'query_text': 'läuft rechtliche gründung gmbh?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v8',
        'query_text': 'läuft rechtliche gründung gmbh Bitte mit Verweisen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse',
        'query_text': 'Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v3',
        'query_text': 'Angenommen, Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v6',
        'query_text': 'wichtige entscheidungen aktiengesellschaft hauptversammlung erforderlich gesetzliche Bestimmungen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v7',
        'query_text': 'wichtige entscheidungen aktiengesellschaft hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v8',
        'query_text': 'wichtige entscheidungen aktiengesellschaft hauptversammlung erforderlich Bitte mit Verweisen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft',
        'query_text': 'Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v3',
        'query_text': 'Angenommen, Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v6',
        'query_text': 'pflichten vorstand aktiengesellschaft gegenüber aktionären gesetzliche Bestimmungen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v7',
        'query_text': 'pflichten vorstand aktiengesellschaft gegenüber aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v8',
        'query_text': 'pflichten vorstand aktiengesellschaft gegenüber aktionären Bitte mit Verweisen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh',
        'query_text': 'Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v3',
        'query_text': 'Angenommen, Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v6',
        'query_text': 'haftung trifft geschäftsführer gmbh pflichtverletzungen gesetzliche Bestimmungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v7',
        'query_text': 'haftung trifft geschäftsführer gmbh pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v8',
        'query_text': 'haftung trifft geschäftsführer gmbh pflichtverletzungen Bitte mit Verweisen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren',
        'query_text': 'Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v3',
        'query_text': 'Angenommen, Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v4',
        'query_text': 'Wie ist läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab gesetzlich geregelt?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v6',
        'query_text': 'Wie läuft ein Verwaltungsverfahren vor einer österreichischen Verwaltungsbehörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v7',
        'query_text': 'läuft verwaltungsverfahren behörde grundsätzlich gesetzliche Bestimmungen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v8',
        'query_text': 'läuft verwaltungsverfahren behörde grundsätzlich?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte',
        'query_text': 'Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v6',
        'query_text': 'rechte habe baugenehmigung verweigert gesetzliche Bestimmungen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v7',
        'query_text': 'rechte habe baugenehmigung verweigert?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v8',
        'query_text': 'rechte habe baugenehmigung verweigert Bitte mit Verweisen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht',
        'query_text': 'Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v3',
        'query_text': 'Angenommen, Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v4',
        'query_text': 'Wie ist kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen gesetzlich geregelt?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v6',
        'query_text': 'bescheid magistrats beschwerde beim verwaltungsgericht einlegen gesetzliche Bestimmungen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v7',
        'query_text': 'bescheid magistrats beschwerde beim verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v8',
        'query_text': 'bescheid magistrats beschwerde beim verwaltungsgericht einlegen Bitte mit Verweisen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid',
        'query_text': 'Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v6',
        'query_text': 'fristen gelten beschwerde verwaltungsbescheid gesetzliche Bestimmungen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v7',
        'query_text': 'fristen gelten beschwerde verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v8',
        'query_text': 'fristen gelten beschwerde verwaltungsbescheid Bitte mit Verweisen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren',
        'query_text': 'Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v3',
        'query_text': 'Angenommen, Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v6',
        'query_text': 'gewährt akteneinsicht laufenden verwaltungsverfahren gesetzliche Bestimmungen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v7',
        'query_text': 'gewährt akteneinsicht laufenden verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v8',
        'query_text': 'gewährt akteneinsicht laufenden verwaltungsverfahren Bitte mit Verweisen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit',
        'query_text': 'Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v3',
        'query_text': 'Angenommen, Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v4',
        'query_text': 'Wie ist die Rechtslage bei Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v6',
        'query_text': 'tun wegen meines geschlechts arbeit benachteiligt werde gesetzliche Bestimmungen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v7',
        'query_text': 'tun wegen meines geschlechts arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v8',
        'query_text': 'tun wegen meines geschlechts arbeit benachteiligt werde Bitte mit Verweisen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft',
        'query_text': 'Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v3',
        'query_text': 'Angenommen, Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v4',
        'query_text': 'Ist es zulässig, dass der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v6',
        'query_text': 'Darf der Dienstgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v7',
        'query_text': 'arbeitgeber wegen meiner ethnischen herkunft einstellen gesetzliche Bestimmungen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v8',
        'query_text': 'arbeitgeber wegen meiner ethnischen herkunft einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job',
        'query_text': 'Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v6',
        'query_text': 'rechte habe wegen meines alters job diskriminiert werde gesetzliche Bestimmungen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v7',
        'query_text': 'rechte habe wegen meines alters job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v8',
        'query_text': 'rechte habe wegen meines alters job diskriminiert werde Bitte mit Verweisen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz',
        'query_text': 'Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v3',
        'query_text': 'Angenommen, Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v6',
        'query_text': 'besonderen schutzbestimmungen gelten jugendliche jahren arbeit gesetzliche Bestimmungen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v7',
        'query_text': 'besonderen schutzbestimmungen gelten jugendliche jahren arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v8',
        'query_text': 'besonderen schutzbestimmungen gelten jugendliche jahren arbeit Bitte mit Verweisen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15',
        'query_text': 'Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v3',
        'query_text': 'Angenommen, Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v4',
        'query_text': 'Wie ist viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten gesetzlich geregelt?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v6',
        'query_text': 'viele stunden 15-jähriger ferien arbeiten gesetzliche Bestimmungen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v7',
        'query_text': 'viele stunden 15-jähriger ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v8',
        'query_text': 'viele stunden 15-jähriger ferien arbeiten Bitte mit Verweisen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen',
        'query_text': 'Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v3',
        'query_text': 'Angenommen, Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v6',
        'query_text': 'schutzfristen gelten geburt werdende mütter gesetzliche Bestimmungen?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v7',
        'query_text': 'schutzfristen gelten geburt werdende mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v8',
        'query_text': 'schutzfristen gelten geburt werdende mütter Bitte mit Verweisen?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung',
        'query_text': 'Darf ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v3',
        'query_text': 'Angenommen, Darf ich während der Elternkarenz geringfügig arbeiten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v4',
        'query_text': 'Ist es zulässig, dass ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v6',
        'query_text': 'elternkarenz geringfügig arbeiten gesetzliche Bestimmungen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v7',
        'query_text': 'elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v8',
        'query_text': 'elternkarenz geringfügig arbeiten Bitte mit Verweisen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer',
        'query_text': 'Wie lange kann ich Kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie lange kann ich Kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v3',
        'query_text': 'Angenommen, Wie lange kann ich Kinderbetreuungsgeld beziehen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v4',
        'query_text': 'Wie ist lange kann ich Kinderbetreuungsgeld beziehen gesetzlich geregelt?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie lange kann ich Kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v6',
        'query_text': 'lange kinderbetreuungsgeld beziehen gesetzliche Bestimmungen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v7',
        'query_text': 'lange kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v8',
        'query_text': 'lange kinderbetreuungsgeld beziehen Bitte mit Verweisen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle',
        'query_text': 'Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v3',
        'query_text': 'Angenommen, Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v6',
        'query_text': 'modelle kinderbetreuungsgeldes gibt wähle gesetzliche Bestimmungen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v7',
        'query_text': 'modelle kinderbetreuungsgeldes gibt wähle?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v8',
        'query_text': 'modelle kinderbetreuungsgeldes gibt wähle Bitte mit Verweisen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle',
        'query_text': 'An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v3',
        'query_text': 'Angenommen, An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v4',
        'query_text': 'Wie ist die Rechtslage bei An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v6',
        'query_text': 'stelle wenden oberösterreich sozialhilfe beantragen gesetzliche Bestimmungen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v7',
        'query_text': 'stelle wenden oberösterreich sozialhilfe beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v8',
        'query_text': 'stelle wenden oberösterreich sozialhilfe beantragen Bitte mit Verweisen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung',
        'query_text': 'Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v3',
        'query_text': 'Angenommen, Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v6',
        'query_text': 'mein vermögen berechnung sozialhilfe berücksichtigt gesetzliche Bestimmungen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v7',
        'query_text': 'mein vermögen berechnung sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v8',
        'query_text': 'mein vermögen berechnung sozialhilfe berücksichtigt Bitte mit Verweisen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt',
        'query_text': 'Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v3',
        'query_text': 'Angenommen, Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v6',
        'query_text': 'mache wegen diebstahls strafbar supermarkt etwas mitnehme gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v7',
        'query_text': 'mache wegen diebstahls strafbar supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v8',
        'query_text': 'mache wegen diebstahls strafbar supermarkt etwas mitnehme Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken',
        'query_text': 'Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v3',
        'query_text': 'Angenommen, Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v6',
        'query_text': 'möglichkeiten habe strafzettel falschparken bekommen gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v7',
        'query_text': 'möglichkeiten habe strafzettel falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v8',
        'query_text': 'möglichkeiten habe strafzettel falschparken bekommen habe Bitte mit Verweisen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung',
        'query_text': 'Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v3',
        'query_text': 'Angenommen, Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v4',
        'query_text': 'Ist es möglich, dass ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v6',
        'query_text': 'Kann ich Einspruch gegen eine Organstrafverfügung der Polizeibehörde erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v7',
        'query_text': 'einspruch organstrafverfügung polizei erheben gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v8',
        'query_text': 'einspruch organstrafverfügung polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr',
        'query_text': 'Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v3',
        'query_text': 'Angenommen, Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v6',
        'query_text': 'regeln gelten benutzung e-scootern straßenverkehr gesetzliche Bestimmungen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v7',
        'query_text': 'regeln gelten benutzung e-scootern straßenverkehr?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v8',
        'query_text': 'regeln gelten benutzung e-scootern straßenverkehr Bitte mit Verweisen?',
        'consensus_law': 'StVO',
    },
]
TRAIN_QUERY_SET: List[Dict[str, str]] = [
    {
        'query_id': 'Q1_arbeitgeber_haftung',
        'query_text': 'Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v3',
        'query_text': 'Angenommen, Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v4',
        'query_text': 'Wann haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Unter welchen Voraussetzungen haftet der Arbeitgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v6',
        'query_text': 'Unter welchen Voraussetzungen haftet der Dienstgeber für Schäden des Arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen',
        'query_text': 'Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v3',
        'query_text': 'Angenommen, Welche Rechte hat der Betriebsrat bei Kündigungen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte hat der Betriebsrat bei Kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v6',
        'query_text': 'rechte betriebsrat kündigungen gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv',
        'query_text': 'In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v3',
        'query_text': 'Angenommen, In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v4',
        'query_text': 'Wie ist die Rechtslage bei In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: In welchen Fällen haftet der Dienstgeber gegenüber den Trägern der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v6',
        'query_text': 'welchen fällen haftet dienstgeber gegenüber trägern sozialversicherung gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren',
        'query_text': 'Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Beschuldigter in einem Strafverfahren. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Beschuldigter in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v6',
        'query_text': 'rechte habe beschuldigter strafverfahren gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt',
        'query_text': 'Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v3',
        'query_text': 'Angenommen, Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v4',
        'query_text': 'Wie ist kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt gesetzlich geregelt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich gegen jemanden klagen, der mir Geld schuldet und nicht zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v6',
        'query_text': 'jemanden klagen geld schuldet zahlt gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche',
        'query_text': 'Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v3',
        'query_text': 'Angenommen, Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v4',
        'query_text': 'Ist es zulässig, dass der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Arbeitgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v6',
        'query_text': 'Darf der Dienstgeber mich kündigen, weil ich meine Ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16',
        'query_text': 'Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v3',
        'query_text': 'Angenommen, Was darf ich tun, wenn ich 16 bin. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v4',
        'query_text': 'Wie ist die Rechtslage bei Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was darf ich tun, wenn ich 16 bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v6',
        'query_text': 'tun bin gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung',
        'query_text': 'Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v3',
        'query_text': 'Angenommen, Wann darf der Vermieter den Mietvertrag kündigen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann darf der Vermieter den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v6',
        'query_text': 'Wann darf der Bestandgeber den Mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber',
        'query_text': 'Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich nach dem DSG, wenn mein Arbeitgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v6',
        'query_text': 'Welche Rechte habe ich nach dem DSG, wenn mein Dienstgeber meine Daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch',
        'query_text': 'Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v3',
        'query_text': 'Angenommen, Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann habe ich Anspruch auf Sozialhilfe oder Mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v6',
        'query_text': 'habe anspruch sozialhilfe mindestsicherung gesetzliche Bestimmungen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q11_mobbing_arbeit',
        'query_text': 'Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v3',
        'query_text': 'Angenommen, Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann liegt Mobbing am Arbeitsplatz vor und was kann ich dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v6',
        'query_text': 'liegt mobbing arbeitsplatz dagegen tun gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung',
        'query_text': 'Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v3',
        'query_text': 'Angenommen, Ist eine Kündigung wegen Schwangerschaft zulässig. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v4',
        'query_text': 'Wie ist die Rechtslage bei Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ist eine Kündigung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v6',
        'query_text': 'Ist eine Auflösung wegen Schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte',
        'query_text': 'Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Mieter bei einer Mieterhöhung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Mieter bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v6',
        'query_text': 'Welche Rechte habe ich als Bestandnehmer bei einer Mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden',
        'query_text': 'Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v3',
        'query_text': 'Angenommen, Muss ich dem Vermieter jeden Besucher melden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v4',
        'query_text': 'Wie ist die Rechtslage bei Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Muss ich dem Vermieter jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v6',
        'query_text': 'Muss ich dem Bestandgeber jeden Besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q15_berufung_strafurteil',
        'query_text': 'Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für eine Berufung gegen ein Strafurteil. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für eine Berufung gegen ein Strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v6',
        'query_text': 'fristen gelten berufung strafurteil gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber',
        'query_text': 'Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v3',
        'query_text': 'Angenommen, Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v4',
        'query_text': 'Ist es zulässig, dass mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf mein Arbeitgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v6',
        'query_text': 'Darf mein Dienstgeber meine dienstlichen E-Mails ohne meine Zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung',
        'query_text': 'Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v3',
        'query_text': 'Angenommen, Welche Daten darf eine Behörde über mich speichern. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Daten darf eine Behörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v6',
        'query_text': 'Welche Daten darf eine Verwaltungsbehörde über mich speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q18_betrug_strafbar',
        'query_text': 'Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v3',
        'query_text': 'Angenommen, Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann mache ich mich wegen Betrugs strafbar, wenn ich jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v6',
        'query_text': 'mache wegen betrugs strafbar jemanden täusche gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen',
        'query_text': 'Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v3',
        'query_text': 'Angenommen, Was passiert, wenn ich keine Steuern zahle. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v4',
        'query_text': 'Wie ist die Rechtslage bei Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was passiert, wenn ich keine Steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v6',
        'query_text': 'passiert steuern zahle gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall',
        'query_text': 'Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v3',
        'query_text': 'Angenommen, Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Ansprüche habe ich nach einem Verkehrsunfall auf Schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v6',
        'query_text': 'ansprüche habe verkehrsunfall schmerzensgeld gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch',
        'query_text': 'Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v3',
        'query_text': 'Angenommen, Habe ich Anspruch auf Elternkarenz und wie lange. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v4',
        'query_text': 'Wie ist die Rechtslage bei Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Habe ich Anspruch auf Elternkarenz und wie lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v6',
        'query_text': 'habe anspruch elternkarenz lange gesetzliche Bestimmungen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q22_hund_haftung',
        'query_text': 'Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v3',
        'query_text': 'Angenommen, Wer haftet, wenn mein Hund jemanden verletzt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer haftet, wenn mein Hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v6',
        'query_text': 'haftet mein hund jemanden verletzt gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument',
        'query_text': 'Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Konsument bei einem Online-Kauf. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Konsument bei einem Online-Kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v6',
        'query_text': 'rechte habe konsument online-kauf gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft',
        'query_text': 'Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v3',
        'query_text': 'Angenommen, Wann ist eine Videoüberwachung in einem Geschäft zulässig. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann ist eine Videoüberwachung in einem Geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v6',
        'query_text': 'videoüberwachung geschäft zulässig gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung',
        'query_text': 'Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v3',
        'query_text': 'Angenommen, Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v4',
        'query_text': 'Ist es zulässig, dass der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Vermieter die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v6',
        'query_text': 'Darf der Bestandgeber die Kaution einbehalten, wenn nur normale Abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag',
        'query_text': 'Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v3',
        'query_text': 'Angenommen, Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v4',
        'query_text': 'Wie ist die Rechtslage bei Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ab wann ist ein mündlicher Vertrag in Österreich rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v6',
        'query_text': 'mündlicher vertrag rechtsverbindlich gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum',
        'query_text': 'Kann ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v3',
        'query_text': 'Angenommen, Kann ich einen Kaufvertrag wegen Irrtums anfechten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v4',
        'query_text': 'Ist es möglich, dass ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich einen Kaufvertrag wegen Irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v6',
        'query_text': 'kaufvertrag wegen irrtums anfechten gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz',
        'query_text': 'Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v3',
        'query_text': 'Angenommen, Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer haftet, wenn ein Kind bei einem Freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v6',
        'query_text': 'haftet kind freund etwas kaputt macht gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz',
        'query_text': 'Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v3',
        'query_text': 'Angenommen, Wann verjähren Schadenersatzansprüche nach österreichischem Recht. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann verjähren Schadenersatzansprüche nach österreichischem Recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v6',
        'query_text': 'verjähren schadenersatzansprüche recht gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten',
        'query_text': 'Darf ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v3',
        'query_text': 'Angenommen, Darf ich ein gefundenes Handy einfach behalten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v4',
        'query_text': 'Ist es zulässig, dass ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf ich ein gefundenes Handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v6',
        'query_text': 'gefundenes handy einfach behalten gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern',
        'query_text': 'In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v3',
        'query_text': 'Angenommen, In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v4',
        'query_text': 'Wie ist die Rechtslage bei In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: In welchen Fällen habe ich Anspruch auf Unterhalt von meinen Eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v6',
        'query_text': 'welchen fällen habe anspruch unterhalt meinen eltern gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit',
        'query_text': 'Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v3',
        'query_text': 'Angenommen, Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Mitbestimmungsrechte hat der Betriebsrat bei Änderungen der Arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v6',
        'query_text': 'mitbestimmungsrechte betriebsrat änderungen arbeitszeit gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung',
        'query_text': 'Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v3',
        'query_text': 'Angenommen, Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v4',
        'query_text': 'Wie ist wird ein Betriebsrat in einem Unternehmen gegründet und gewählt gesetzlich geregelt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie wird ein Betriebsrat in einem Unternehmen gegründet und gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v6',
        'query_text': 'betriebsrat unternehmen gegründet gewählt gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden',
        'query_text': 'Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v3',
        'query_text': 'Angenommen, Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v4',
        'query_text': 'Ist es zulässig, dass der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Arbeitgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v6',
        'query_text': 'Darf der Dienstgeber ohne Zustimmung des Betriebsrats dauerhaft Überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl',
        'query_text': 'Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Arbeitnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v6',
        'query_text': 'Welche Rechte habe ich als Dienstnehmer bei der Betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv',
        'query_text': 'Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v3',
        'query_text': 'Angenommen, Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Pflichten habe ich als Dienstnehmer gegenüber der Sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v6',
        'query_text': 'pflichten habe dienstnehmer gegenüber sozialversicherung gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung',
        'query_text': 'Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v3',
        'query_text': 'Angenommen, Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v4',
        'query_text': 'Wie ist die Rechtslage bei Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Bin ich als geringfügig Beschäftigter in der Krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v6',
        'query_text': 'bin geringfügig beschäftigter krankenversicherung abgesichert gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten',
        'query_text': 'Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v3',
        'query_text': 'Angenommen, Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer zahlt meine Behandlungskosten nach einem Arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v6',
        'query_text': 'zahlt meine behandlungskosten arbeitsunfall gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch',
        'query_text': 'Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v3',
        'query_text': 'Angenommen, Wann habe ich Anspruch auf eine Invaliditätspension in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann habe ich Anspruch auf eine Invaliditätspension in Österreich?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v6',
        'query_text': 'habe anspruch invaliditätspension gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift',
        'query_text': 'Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v3',
        'query_text': 'Angenommen, Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v4',
        'query_text': 'Ist es möglich, dass der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann der Vermieter die Miete erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v6',
        'query_text': 'Kann der Bestandgeber die Mietzins erhöhen, weil ein neuer Lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen',
        'query_text': 'Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für die Anfechtung des Mietzinses beim Gericht oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v6',
        'query_text': 'Welche Fristen gelten für die Anfechtung des Mietzinses beim Gerichtshof oder bei der Schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot',
        'query_text': 'Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v3',
        'query_text': 'Angenommen, Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v4',
        'query_text': 'Ist es zulässig, dass der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Vermieter mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v6',
        'query_text': 'Darf der Bestandgeber mir Haustiere im Mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte',
        'query_text': 'Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Wohnungseigentümer in der Eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v6',
        'query_text': 'rechte habe wohnungseigentümer eigentümergemeinschaft gesetzliche Bestimmungen?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen',
        'query_text': 'Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v3',
        'query_text': 'Angenommen, Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v4',
        'query_text': 'Wie ist kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen gesetzlich geregelt?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich von einem Unternehmen Auskunft über die von mir gespeicherten Daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v6',
        'query_text': 'unternehmen auskunft gespeicherten daten verlangen gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung',
        'query_text': 'Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v3',
        'query_text': 'Angenommen, Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v4',
        'query_text': 'Wie ist die Rechtslage bei Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Unter welchen Bedingungen darf ein Online-Shop meine Daten zu Werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v6',
        'query_text': 'welchen bedingungen online-shop meine daten werbezwecken verwenden gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter',
        'query_text': 'Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v3',
        'query_text': 'Angenommen, Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v4',
        'query_text': 'Ist es zulässig, dass mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf mein Internetanbieter meine Verbindungsdaten speichern und auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v6',
        'query_text': 'mein internetanbieter meine verbindungsdaten speichern auswerten gesetzliche Bestimmungen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite',
        'query_text': 'Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v3',
        'query_text': 'Angenommen, Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Regeln gelten für Cookies und Tracking auf Webseiten in Österreich?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v6',
        'query_text': 'regeln gelten cookies tracking webseiten gesetzliche Bestimmungen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene',
        'query_text': 'Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich, wenn meine personenbezogenen Daten bei einem Datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v6',
        'query_text': 'rechte habe meine personenbezogenen daten datenleck veröffentlicht wurden gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website',
        'query_text': 'Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v3',
        'query_text': 'Angenommen, Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v4',
        'query_text': 'Ist es zulässig, dass eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf eine Schule Fotos von mir ohne meine Zustimmung auf ihrer Website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v6',
        'query_text': 'schule fotos meine zustimmung ihrer website veröffentlichen gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten',
        'query_text': 'Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v3',
        'query_text': 'Angenommen, Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche strafrechtlichen Folgen drohen, wenn ich ohne Fahrschein fahre und der Kontrolle falsche Daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v6',
        'query_text': 'strafrechtlichen folgen drohen fahrschein fahre kontrolle falsche daten angebe gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar',
        'query_text': 'Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v3',
        'query_text': 'Angenommen, Ab wann mache ich mich wegen Körperverletzung strafbar. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v4',
        'query_text': 'Wie ist die Rechtslage bei Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ab wann mache ich mich wegen Körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v6',
        'query_text': 'mache wegen körperverletzung strafbar gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei',
        'query_text': 'Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v3',
        'query_text': 'Angenommen, Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v4',
        'query_text': 'Ist es zulässig, dass die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf die Polizei mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v6',
        'query_text': 'Darf die Polizeibehörde mein Handy ohne richterlichen Beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer',
        'query_text': 'Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v3',
        'query_text': 'Angenommen, Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v4',
        'query_text': 'Wie ist lange darf ich in Österreich in Untersuchungshaft gehalten werden gesetzlich geregelt?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie lange darf ich in Österreich in Untersuchungshaft gehalten werden?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v6',
        'query_text': 'lange untersuchungshaft gehalten gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren',
        'query_text': 'Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Opfer in einem Strafverfahren. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Opfer in einem Strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v6',
        'query_text': 'rechte habe opfer strafverfahren gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen',
        'query_text': 'Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v3',
        'query_text': 'Angenommen, Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v4',
        'query_text': 'Wie ist die Rechtslage bei Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was passiert, wenn ich eine Verwaltungsstrafe nicht bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v6',
        'query_text': 'passiert verwaltungsstrafe bezahle gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer',
        'query_text': 'Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v3',
        'query_text': 'Angenommen, Welche Strafen drohen bei Trunkenheit am Steuer in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Strafen drohen bei Trunkenheit am Steuer in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v6',
        'query_text': 'strafen drohen trunkenheit steuer gesetzliche Bestimmungen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung',
        'query_text': 'Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v3',
        'query_text': 'Angenommen, Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann verjährt eine Verwaltungsübertretung nach österreichischem Recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v6',
        'query_text': 'verjährt verwaltungsübertretung recht gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer',
        'query_text': 'Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für die Abgabe der Einkommensteuererklärung beim Finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v6',
        'query_text': 'fristen gelten abgabe einkommensteuererklärung beim finanzamt gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid',
        'query_text': 'Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v3',
        'query_text': 'Angenommen, Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v4',
        'query_text': 'Wie ist kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen gesetzlich geregelt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich gegen einen Einkommensteuerbescheid Beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v6',
        'query_text': 'einkommensteuerbescheid beschwerde einlegen gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen',
        'query_text': 'Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v3',
        'query_text': 'Angenommen, Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v4',
        'query_text': 'Wie ist die Rechtslage bei Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was passiert, wenn ich Umsätze in meiner Steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v6',
        'query_text': 'passiert umsätze meiner steuererklärung verschweige gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern',
        'query_text': 'Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v3',
        'query_text': 'Angenommen, Muss ich in Österreich auch Auslandseinkünfte versteuern. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v4',
        'query_text': 'Wie ist die Rechtslage bei Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Muss ich in Österreich auch Auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v6',
        'query_text': 'auslandseinkünfte versteuern gesetzliche Bestimmungen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft',
        'query_text': 'Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v3',
        'query_text': 'Angenommen, Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v4',
        'query_text': 'Ist es möglich, dass ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich einen im Geschäft gekauften Artikel ohne Angabe von Gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v6',
        'query_text': 'geschäft gekauften artikel angabe gründen zurückgeben gesetzliche Bestimmungen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung',
        'query_text': 'Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v3',
        'query_text': 'Angenommen, Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Widerrufsfrist gilt bei Online-Bestellungen in Österreich?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v6',
        'query_text': 'widerrufsfrist gilt online-bestellungen gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware',
        'query_text': 'Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v3',
        'query_text': 'Angenommen, Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v4',
        'query_text': 'Ist es zulässig, dass ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf ein Unternehmer die Gewährleistung bei gebrauchten Waren ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v6',
        'query_text': 'unternehmer gewährleistung gebrauchten ausschließen gesetzliche Bestimmungen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung',
        'query_text': 'Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v3',
        'query_text': 'Angenommen, Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v4',
        'query_text': 'Wie ist die Rechtslage bei Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was sind meine Rechte, wenn ein Online-Abonnement automatisch verlängert wird?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v6',
        'query_text': 'meine rechte online-abonnement automatisch verlängert gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten',
        'query_text': 'Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v3',
        'query_text': 'Angenommen, Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Informationen muss mir ein Unternehmer vor Vertragsabschluss im Fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v6',
        'query_text': 'informationen unternehmer vertragsabschluss fernabsatz geben gesetzliche Bestimmungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage',
        'query_text': 'Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v3',
        'query_text': 'Angenommen, Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v4',
        'query_text': 'Wie ist die Rechtslage bei Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Vor welchem Gericht muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v6',
        'query_text': 'Vor welchem Gerichtshof muss ich klagen, wenn der Schuldner in einem anderen Bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt',
        'query_text': 'Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v3',
        'query_text': 'Angenommen, Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Kosten fallen bei einer Zivilklage an und wer trägt sie im Erfolgs- oder Misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v6',
        'query_text': 'kosten fallen zivilklage trägt erfolgs- misserfolgsfall gesetzliche Bestimmungen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen',
        'query_text': 'Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v3',
        'query_text': 'Angenommen, Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v4',
        'query_text': 'Ist es möglich, dass ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich Verfahrenshilfe beantragen, wenn ich mir einen Zivilprozess nicht leisten kann?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v6',
        'query_text': 'verfahrenshilfe beantragen zivilprozess leisten gesetzliche Bestimmungen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung',
        'query_text': 'Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v3',
        'query_text': 'Angenommen, Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer haftet, wenn ich mit meinem Auto einen Fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v6',
        'query_text': 'haftet meinem auto fußgänger verletze gesetzliche Bestimmungen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung',
        'query_text': 'Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v3',
        'query_text': 'Angenommen, Haftet der Halter oder der Lenker bei einem Autounfall. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v4',
        'query_text': 'Wie ist die Rechtslage bei Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Haftet der Halter oder der Lenker bei einem Autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v6',
        'query_text': 'haftet halter lenker autounfall gesetzliche Bestimmungen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang',
        'query_text': 'Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v3',
        'query_text': 'Angenommen, Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v4',
        'query_text': 'Ist es möglich, dass ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich nach einem Verkehrsunfall auch Verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v6',
        'query_text': 'verkehrsunfall verdienstentgang geltend machen gesetzliche Bestimmungen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo',
        'query_text': 'Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v3',
        'query_text': 'Angenommen, Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Pflichten habe ich bei einem Unfall mit Sachschaden nach der Straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v6',
        'query_text': 'pflichten habe unfall sachschaden straßenverkehrsordnung gesetzliche Bestimmungen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung',
        'query_text': 'Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v3',
        'query_text': 'Angenommen, Ab wann muss ich als Unternehmer doppelte Buchhaltung führen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v4',
        'query_text': 'Wie ist die Rechtslage bei Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Ab wann muss ich als Unternehmer doppelte Buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v6',
        'query_text': 'unternehmer doppelte buchhaltung führen gesetzliche Bestimmungen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q75_impressumspflicht_website',
        'query_text': 'Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v3',
        'query_text': 'Angenommen, Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Angaben muss ein Impressum auf einer Unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v6',
        'query_text': 'angaben impressum unternehmenswebsite enthalten gesetzliche Bestimmungen?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung',
        'query_text': 'Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich als Gesellschafter einer GmbH auf Einsicht in die Unterlagen der Gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v6',
        'query_text': 'rechte habe gesellschafter gmbh einsicht unterlagen gesellschaft gesetzliche Bestimmungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf',
        'query_text': 'Wie läuft die rechtliche Gründung einer GmbH in Österreich ab?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie läuft die rechtliche Gründung einer GmbH in Österreich ab?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v3',
        'query_text': 'Angenommen, Wie läuft die rechtliche Gründung einer GmbH in Österreich ab. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v4',
        'query_text': 'Wie ist läuft die rechtliche Gründung einer GmbH in Österreich ab gesetzlich geregelt?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie läuft die rechtliche Gründung einer GmbH in Österreich ab?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v6',
        'query_text': 'läuft rechtliche gründung gmbh gesetzliche Bestimmungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse',
        'query_text': 'Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v3',
        'query_text': 'Angenommen, Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann ist für wichtige Entscheidungen einer Aktiengesellschaft eine Hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v6',
        'query_text': 'wichtige entscheidungen aktiengesellschaft hauptversammlung erforderlich gesetzliche Bestimmungen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft',
        'query_text': 'Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v3',
        'query_text': 'Angenommen, Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Pflichten hat der Vorstand einer Aktiengesellschaft gegenüber den Aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v6',
        'query_text': 'pflichten vorstand aktiengesellschaft gegenüber aktionären gesetzliche Bestimmungen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh',
        'query_text': 'Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v3',
        'query_text': 'Angenommen, Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Haftung trifft mich als Geschäftsführer einer GmbH bei Pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v6',
        'query_text': 'haftung trifft geschäftsführer gmbh pflichtverletzungen gesetzliche Bestimmungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren',
        'query_text': 'Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v3',
        'query_text': 'Angenommen, Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v4',
        'query_text': 'Wie ist läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab gesetzlich geregelt?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie läuft ein Verwaltungsverfahren vor einer österreichischen Behörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v6',
        'query_text': 'Wie läuft ein Verwaltungsverfahren vor einer österreichischen Verwaltungsbehörde grundsätzlich ab?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte',
        'query_text': 'Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich, wenn mir eine Baugenehmigung verweigert wird?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v6',
        'query_text': 'rechte habe baugenehmigung verweigert gesetzliche Bestimmungen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht',
        'query_text': 'Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v3',
        'query_text': 'Angenommen, Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v4',
        'query_text': 'Wie ist kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen gesetzlich geregelt?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie kann ich gegen einen Bescheid eines Magistrats Beschwerde beim Verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v6',
        'query_text': 'bescheid magistrats beschwerde beim verwaltungsgericht einlegen gesetzliche Bestimmungen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid',
        'query_text': 'Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v3',
        'query_text': 'Angenommen, Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Fristen gelten für eine Beschwerde gegen einen Verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v6',
        'query_text': 'fristen gelten beschwerde verwaltungsbescheid gesetzliche Bestimmungen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren',
        'query_text': 'Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v3',
        'query_text': 'Angenommen, Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v4',
        'query_text': 'Wie ist die Rechtslage bei Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wer gewährt mir Akteneinsicht in einem laufenden Verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v6',
        'query_text': 'gewährt akteneinsicht laufenden verwaltungsverfahren gesetzliche Bestimmungen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit',
        'query_text': 'Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v3',
        'query_text': 'Angenommen, Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v4',
        'query_text': 'Wie ist die Rechtslage bei Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Was kann ich tun, wenn ich wegen meines Geschlechts bei der Arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v6',
        'query_text': 'tun wegen meines geschlechts arbeit benachteiligt werde gesetzliche Bestimmungen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft',
        'query_text': 'Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v3',
        'query_text': 'Angenommen, Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v4',
        'query_text': 'Ist es zulässig, dass der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf der Arbeitgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v6',
        'query_text': 'Darf der Dienstgeber mich wegen meiner ethnischen Herkunft nicht einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job',
        'query_text': 'Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v3',
        'query_text': 'Angenommen, Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Rechte habe ich, wenn ich wegen meines Alters im Job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v6',
        'query_text': 'rechte habe wegen meines alters job diskriminiert werde gesetzliche Bestimmungen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz',
        'query_text': 'Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v3',
        'query_text': 'Angenommen, Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche besonderen Schutzbestimmungen gelten für Jugendliche unter 18 Jahren in der Arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v6',
        'query_text': 'besonderen schutzbestimmungen gelten jugendliche jahren arbeit gesetzliche Bestimmungen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15',
        'query_text': 'Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v3',
        'query_text': 'Angenommen, Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v4',
        'query_text': 'Wie ist viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten gesetzlich geregelt?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie viele Stunden darf ich als 15-Jähriger in den Ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v6',
        'query_text': 'viele stunden 15-jähriger ferien arbeiten gesetzliche Bestimmungen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen',
        'query_text': 'Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v3',
        'query_text': 'Angenommen, Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Schutzfristen gelten vor und nach der Geburt für werdende Mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v6',
        'query_text': 'schutzfristen gelten geburt werdende mütter gesetzliche Bestimmungen?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung',
        'query_text': 'Darf ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Darf ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v3',
        'query_text': 'Angenommen, Darf ich während der Elternkarenz geringfügig arbeiten. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v4',
        'query_text': 'Ist es zulässig, dass ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Darf ich während der Elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v6',
        'query_text': 'elternkarenz geringfügig arbeiten gesetzliche Bestimmungen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer',
        'query_text': 'Wie lange kann ich Kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wie lange kann ich Kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v3',
        'query_text': 'Angenommen, Wie lange kann ich Kinderbetreuungsgeld beziehen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v4',
        'query_text': 'Wie ist lange kann ich Kinderbetreuungsgeld beziehen gesetzlich geregelt?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wie lange kann ich Kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v6',
        'query_text': 'lange kinderbetreuungsgeld beziehen gesetzliche Bestimmungen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle',
        'query_text': 'Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v3',
        'query_text': 'Angenommen, Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Modelle des Kinderbetreuungsgeldes gibt es und wie wähle ich eines aus?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v6',
        'query_text': 'modelle kinderbetreuungsgeldes gibt wähle gesetzliche Bestimmungen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle',
        'query_text': 'An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v3',
        'query_text': 'Angenommen, An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v4',
        'query_text': 'Wie ist die Rechtslage bei An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: An welche Stelle muss ich mich wenden, um in Oberösterreich Sozialhilfe zu beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v6',
        'query_text': 'stelle wenden oberösterreich sozialhilfe beantragen gesetzliche Bestimmungen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung',
        'query_text': 'Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v3',
        'query_text': 'Angenommen, Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v4',
        'query_text': 'Wie ist die Rechtslage bei Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wird mein Vermögen bei der Berechnung der Sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v6',
        'query_text': 'mein vermögen berechnung sozialhilfe berücksichtigt gesetzliche Bestimmungen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt',
        'query_text': 'Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v3',
        'query_text': 'Angenommen, Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v4',
        'query_text': 'Wie ist die Rechtslage bei Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Wann mache ich mich wegen Diebstahls strafbar, wenn ich im Supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v6',
        'query_text': 'mache wegen diebstahls strafbar supermarkt etwas mitnehme gesetzliche Bestimmungen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken',
        'query_text': 'Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v3',
        'query_text': 'Angenommen, Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Möglichkeiten habe ich, wenn ich einen Strafzettel für Falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v6',
        'query_text': 'möglichkeiten habe strafzettel falschparken bekommen gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung',
        'query_text': 'Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v3',
        'query_text': 'Angenommen, Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v4',
        'query_text': 'Ist es möglich, dass ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Kann ich Einspruch gegen eine Organstrafverfügung der Polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v6',
        'query_text': 'Kann ich Einspruch gegen eine Organstrafverfügung der Polizeibehörde erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr',
        'query_text': 'Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v2',
        'query_text': 'Welche gesetzliche Grundlage und Voraussetzungen sind einschlägig für Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v3',
        'query_text': 'Angenommen, Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich. Welche Rechtsfolgen ergeben sich daraus?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v4',
        'query_text': 'Welche gesetzlichen Vorgaben gelten für Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v5',
        'query_text': 'Wie ist folgender Sachverhalt rechtlich zu beurteilen: Welche Regeln gelten für die Benutzung von E-Scootern im Straßenverkehr in Österreich?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v6',
        'query_text': 'regeln gelten benutzung e-scootern straßenverkehr gesetzliche Bestimmungen?',
        'consensus_law': 'StVO',
    },
]
TEST_QUERY_SET: List[Dict[str, str]] = [
    {
        'query_id': 'Q1_arbeitgeber_haftung_v7',
        'query_text': 'welchen voraussetzungen haftet arbeitgeber schäden arbeitnehmers gesetzliche Bestimmungen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q1_arbeitgeber_haftung_v8',
        'query_text': 'welchen voraussetzungen haftet arbeitgeber schäden arbeitnehmers?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v7',
        'query_text': 'rechte betriebsrat kündigungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q2_betriebsrat_kuendigungen_v8',
        'query_text': 'rechte betriebsrat kündigungen Bitte mit Verweisen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v7',
        'query_text': 'welchen fällen haftet dienstgeber gegenüber trägern sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q3_dienstgeber_sv_v8',
        'query_text': 'welchen fällen haftet dienstgeber gegenüber trägern sozialversicherung Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v7',
        'query_text': 'rechte habe beschuldigter strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q4_beschuldigter_strafverfahren_v8',
        'query_text': 'rechte habe beschuldigter strafverfahren Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v7',
        'query_text': 'jemanden klagen geld schuldet zahlt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q5_geld_schuldet_nicht_zahlt_v8',
        'query_text': 'jemanden klagen geld schuldet zahlt Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v7',
        'query_text': 'arbeitgeber kündigen meine ansprüche geltend mache gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q6_kuendigung_ansprueche_v8',
        'query_text': 'arbeitgeber kündigen meine ansprüche geltend mache?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v7',
        'query_text': 'tun bin?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q7_was_darf_ich_mit_16_v8',
        'query_text': 'tun bin Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v7',
        'query_text': 'vermieter mietvertrag kündigen gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q8_mietrecht_kuendigung_v8',
        'query_text': 'vermieter mietvertrag kündigen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v7',
        'query_text': 'rechte habe dsg mein arbeitgeber meine daten speichert gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q9_datenschutz_arbeitgeber_v8',
        'query_text': 'rechte habe dsg mein arbeitgeber meine daten speichert?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v7',
        'query_text': 'habe anspruch sozialhilfe mindestsicherung?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q10_sozialhilfe_anspruch_v8',
        'query_text': 'habe anspruch sozialhilfe mindestsicherung Bitte mit Verweisen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v7',
        'query_text': 'liegt mobbing arbeitsplatz dagegen tun?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q11_mobbing_arbeit_v8',
        'query_text': 'liegt mobbing arbeitsplatz dagegen tun Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v7',
        'query_text': 'kündigung wegen schwangerschaft zulässig gesetzliche Bestimmungen?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q12_schwangerschaft_kuendigung_v8',
        'query_text': 'kündigung wegen schwangerschaft zulässig?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v7',
        'query_text': 'rechte habe mieter mieterhöhung gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q13_mieterhoehung_rechte_v8',
        'query_text': 'rechte habe mieter mieterhöhung?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v7',
        'query_text': 'vermieter jeden besucher melden gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q14_besucher_melden_v8',
        'query_text': 'vermieter jeden besucher melden?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v7',
        'query_text': 'fristen gelten berufung strafurteil?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q15_berufung_strafurteil_v8',
        'query_text': 'fristen gelten berufung strafurteil Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v7',
        'query_text': 'mein arbeitgeber meine dienstlichen e-mails zustimmung lesen gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q16_email_ueberwachung_arbeitgeber_v8',
        'query_text': 'mein arbeitgeber meine dienstlichen e-mails meine zustimmung lesen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v7',
        'query_text': 'daten behörde speichern gesetzliche Bestimmungen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q17_behoerde_datenspeicherung_v8',
        'query_text': 'daten behörde speichern?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v7',
        'query_text': 'mache wegen betrugs strafbar jemanden täusche?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q18_betrug_strafbar_v8',
        'query_text': 'mache wegen betrugs strafbar jemanden täusche Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v7',
        'query_text': 'passiert steuern zahle?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q19_steuern_nicht_zahlen_v8',
        'query_text': 'passiert steuern zahle Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v7',
        'query_text': 'ansprüche habe verkehrsunfall schmerzensgeld?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q20_schmerzensgeld_verkehrsunfall_v8',
        'query_text': 'ansprüche habe verkehrsunfall schmerzensgeld Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v7',
        'query_text': 'habe anspruch elternkarenz lange?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q21_elternkarenz_anspruch_v8',
        'query_text': 'habe anspruch elternkarenz lange Bitte mit Verweisen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q22_hund_haftung_v7',
        'query_text': 'haftet mein hund jemanden verletzt?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q22_hund_haftung_v8',
        'query_text': 'haftet mein hund jemanden verletzt Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v7',
        'query_text': 'rechte habe konsument online-kauf?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q23_onlinekauf_konsument_v8',
        'query_text': 'rechte habe konsument online-kauf Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v7',
        'query_text': 'videoüberwachung geschäft zulässig?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q24_videoueberwachung_geschaeft_v8',
        'query_text': 'videoüberwachung geschäft zulässig Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v7',
        'query_text': 'vermieter kaution einbehalten nur normale abnutzung vorliegt gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q25_kaution_normal_abnutzung_v8',
        'query_text': 'vermieter kaution einbehalten nur normale abnutzung vorliegt?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v7',
        'query_text': 'mündlicher vertrag rechtsverbindlich?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q26_muendlicher_vertrag_v8',
        'query_text': 'mündlicher vertrag rechtsverbindlich Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v7',
        'query_text': 'kaufvertrag wegen irrtums anfechten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q27_kaufvertrag_irrtum_v8',
        'query_text': 'kaufvertrag wegen irrtums anfechten Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v7',
        'query_text': 'haftet kind freund etwas kaputt macht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q28_kind_schadenersatz_v8',
        'query_text': 'haftet kind freund etwas kaputt macht Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v7',
        'query_text': 'verjähren schadenersatzansprüche recht?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q29_verjaehrung_schadenersatz_v8',
        'query_text': 'verjähren schadenersatzansprüche recht Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v7',
        'query_text': 'gefundenes handy einfach behalten?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q30_fund_sache_behalten_v8',
        'query_text': 'gefundenes handy einfach behalten Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v7',
        'query_text': 'welchen fällen habe anspruch unterhalt meinen eltern?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q31_unterhalt_eltern_v8',
        'query_text': 'welchen fällen habe anspruch unterhalt meinen eltern Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v7',
        'query_text': 'mitbestimmungsrechte betriebsrat änderungen arbeitszeit?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q32_betriebsrat_arbeitszeit_v8',
        'query_text': 'mitbestimmungsrechte betriebsrat änderungen arbeitszeit Bitte mit Verweisen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v7',
        'query_text': 'betriebsrat unternehmen gegründet gewählt?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q33_betriebsrat_gruendung_v8',
        'query_text': 'betriebsrat unternehmen gegründet gewählt Bitte mit Verweisen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v7',
        'query_text': 'arbeitgeber zustimmung betriebsrats dauerhaft überstunden anordnen gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q34_betriebsrat_ueberstunden_v8',
        'query_text': 'arbeitgeber zustimmung betriebsrats dauerhaft überstunden anordnen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v7',
        'query_text': 'rechte habe arbeitnehmer betriebsratswahl gesetzliche Bestimmungen?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q35_rechte_arbeitnehmer_betriebsratswahl_v8',
        'query_text': 'rechte habe arbeitnehmer betriebsratswahl?',
        'consensus_law': 'ArbVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v7',
        'query_text': 'pflichten habe dienstnehmer gegenüber sozialversicherung?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q36_pflichten_dienstnehmer_sv_v8',
        'query_text': 'pflichten habe dienstnehmer gegenüber sozialversicherung Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v7',
        'query_text': 'bin geringfügig beschäftigter krankenversicherung abgesichert?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q37_geringfuegig_krankenversicherung_v8',
        'query_text': 'bin geringfügig beschäftigter krankenversicherung abgesichert Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v7',
        'query_text': 'zahlt meine behandlungskosten arbeitsunfall?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q38_arbeitsunfall_kosten_v8',
        'query_text': 'zahlt meine behandlungskosten arbeitsunfall Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v7',
        'query_text': 'habe anspruch invaliditätspension?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q39_invaliditaetspension_anspruch_v8',
        'query_text': 'habe anspruch invaliditätspension Bitte mit Verweisen?',
        'consensus_law': 'ASVG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v7',
        'query_text': 'vermieter miete erhöhen neuer lift eingebaut wurde gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q40_mieterhoehung_lift_v8',
        'query_text': 'vermieter miete erhöhen neuer lift eingebaut wurde?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v7',
        'query_text': 'fristen gelten anfechtung mietzinses beim gericht schlichtungsstelle gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q41_mietzins_anfechtung_fristen_v8',
        'query_text': 'fristen gelten anfechtung mietzinses beim gericht schlichtungsstelle?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v7',
        'query_text': 'vermieter haustiere mietvertrag generell verbieten gesetzliche Bestimmungen?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q42_haustiere_verbot_v8',
        'query_text': 'vermieter haustiere mietvertrag generell verbieten?',
        'consensus_law': 'MRG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v7',
        'query_text': 'rechte habe wohnungseigentümer eigentümergemeinschaft?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q43_wohnungseigentum_rechte_v8',
        'query_text': 'rechte habe wohnungseigentümer eigentümergemeinschaft Bitte mit Verweisen?',
        'consensus_law': 'WEG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v7',
        'query_text': 'unternehmen auskunft gespeicherten daten verlangen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q44_auskunftsrecht_unternehmen_v8',
        'query_text': 'unternehmen auskunft gespeicherten daten verlangen Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v7',
        'query_text': 'welchen bedingungen online-shop meine daten werbezwecken verwenden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q45_datenverwendung_werbung_v8',
        'query_text': 'welchen bedingungen online-shop meine daten werbezwecken verwenden Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v7',
        'query_text': 'mein internetanbieter meine verbindungsdaten speichern auswerten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q46_verbindungsdaten_internetanbieter_v8',
        'query_text': 'mein internetanbieter meine verbindungsdaten speichern auswerten Bitte mit Verweisen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v7',
        'query_text': 'regeln gelten cookies tracking webseiten?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q47_cookies_tracking_webseite_v8',
        'query_text': 'regeln gelten cookies tracking webseiten Bitte mit Verweisen?',
        'consensus_law': 'TKG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v7',
        'query_text': 'rechte habe meine personenbezogenen daten datenleck veröffentlicht wurden?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q48_datenleck_rechte_betroffene_v8',
        'query_text': 'rechte habe meine personenbezogenen daten datenleck veröffentlicht wurden Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v7',
        'query_text': 'schule fotos meine zustimmung ihrer website veröffentlichen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q49_schule_fotos_website_v8',
        'query_text': 'schule fotos meine zustimmung ihrer website veröffentlichen Bitte mit Verweisen?',
        'consensus_law': 'DSG',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v7',
        'query_text': 'strafrechtlichen folgen drohen fahrschein fahre kontrolle falsche daten angebe?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q50_schwarzfahren_falsche_daten_v8',
        'query_text': 'strafrechtlichen folgen drohen fahrschein fahre kontrolle falsche daten angebe Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v7',
        'query_text': 'mache wegen körperverletzung strafbar?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q51_koerperverletzung_strafbar_v8',
        'query_text': 'mache wegen körperverletzung strafbar Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v7',
        'query_text': 'polizei mein handy richterlichen beschluss durchsuchen gesetzliche Bestimmungen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q52_handy_durchsuchung_polizei_v8',
        'query_text': 'polizei mein handy richterlichen beschluss durchsuchen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v7',
        'query_text': 'lange untersuchungshaft gehalten?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q53_untersuchungshaft_dauer_v8',
        'query_text': 'lange untersuchungshaft gehalten Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v7',
        'query_text': 'rechte habe opfer strafverfahren?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q54_opferrechte_strafverfahren_v8',
        'query_text': 'rechte habe opfer strafverfahren Bitte mit Verweisen?',
        'consensus_law': 'StPO',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v7',
        'query_text': 'passiert verwaltungsstrafe bezahle?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q55_verwaltungsstrafe_nicht_bezahlen_v8',
        'query_text': 'passiert verwaltungsstrafe bezahle Bitte mit Verweisen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v7',
        'query_text': 'strafen drohen trunkenheit steuer?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q56_trunkenheit_am_steuer_v8',
        'query_text': 'strafen drohen trunkenheit steuer Bitte mit Verweisen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v7',
        'query_text': 'verjährt verwaltungsübertretung recht?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q57_verjaehrung_verwaltungsuebertretung_v8',
        'query_text': 'verjährt verwaltungsübertretung recht Bitte mit Verweisen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v7',
        'query_text': 'fristen gelten abgabe einkommensteuererklärung beim finanzamt?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q58_fristen_einkommensteuer_v8',
        'query_text': 'fristen gelten abgabe einkommensteuererklärung beim finanzamt Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v7',
        'query_text': 'einkommensteuerbescheid beschwerde einlegen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q59_beschwerde_steuerbescheid_v8',
        'query_text': 'einkommensteuerbescheid beschwerde einlegen Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v7',
        'query_text': 'passiert umsätze meiner steuererklärung verschweige?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q60_steuerhinterziehung_folgen_v8',
        'query_text': 'passiert umsätze meiner steuererklärung verschweige Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v7',
        'query_text': 'auslandseinkünfte versteuern?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q61_auslandseinkuenfte_versteuern_v8',
        'query_text': 'auslandseinkünfte versteuern Bitte mit Verweisen?',
        'consensus_law': 'BAO',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v7',
        'query_text': 'geschäft gekauften artikel angabe gründen zurückgeben?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q62_rueckgabe_ware_geschaeft_v8',
        'query_text': 'geschäft gekauften artikel angabe gründen zurückgeben Bitte mit Verweisen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v7',
        'query_text': 'widerrufsfrist gilt online-bestellungen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q63_widerrufsfrist_onlinebestellung_v8',
        'query_text': 'widerrufsfrist gilt online-bestellungen Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v7',
        'query_text': 'unternehmer gewährleistung gebrauchten ausschließen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q64_gewaehrleistung_gebrauchtware_v8',
        'query_text': 'unternehmer gewährleistung gebrauchten ausschließen Bitte mit Verweisen?',
        'consensus_law': 'KSchG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v7',
        'query_text': 'meine rechte online-abonnement automatisch verlängert?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q65_autoabo_verlaengerung_v8',
        'query_text': 'meine rechte online-abonnement automatisch verlängert Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v7',
        'query_text': 'informationen unternehmer vertragsabschluss fernabsatz geben?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q66_fernabsatz_informationspflichten_v8',
        'query_text': 'informationen unternehmer vertragsabschluss fernabsatz geben Bitte mit Verweisen?',
        'consensus_law': 'FAGG',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v7',
        'query_text': 'welchem gericht klagen schuldner anderen bundesland wohnt gesetzliche Bestimmungen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q67_oertliche_zustaendigkeit_klage_v8',
        'query_text': 'welchem gericht klagen schuldner anderen bundesland wohnt?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v7',
        'query_text': 'kosten fallen zivilklage trägt erfolgs- misserfolgsfall?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q68_prozesskosten_wer_traegt_v8',
        'query_text': 'kosten fallen zivilklage trägt erfolgs- misserfolgsfall Bitte mit Verweisen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v7',
        'query_text': 'verfahrenshilfe beantragen zivilprozess leisten?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q69_verfahrenshilfe_beanspruchen_v8',
        'query_text': 'verfahrenshilfe beantragen zivilprozess leisten Bitte mit Verweisen?',
        'consensus_law': 'ZPO',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v7',
        'query_text': 'haftet meinem auto fußgänger verletze?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q70_auto_fussgaenger_haftung_v8',
        'query_text': 'haftet meinem auto fußgänger verletze Bitte mit Verweisen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v7',
        'query_text': 'haftet halter lenker autounfall?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q71_halter_lenker_haftung_v8',
        'query_text': 'haftet halter lenker autounfall Bitte mit Verweisen?',
        'consensus_law': 'EKHG',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v7',
        'query_text': 'verkehrsunfall verdienstentgang geltend machen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q72_verkehrsunfall_verdienstentgang_v8',
        'query_text': 'verkehrsunfall verdienstentgang geltend machen Bitte mit Verweisen?',
        'consensus_law': 'ABGB',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v7',
        'query_text': 'pflichten habe unfall sachschaden straßenverkehrsordnung?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q73_unfall_pflichten_stvo_v8',
        'query_text': 'pflichten habe unfall sachschaden straßenverkehrsordnung Bitte mit Verweisen?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v7',
        'query_text': 'unternehmer doppelte buchhaltung führen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q74_doppelte_buchhaltung_v8',
        'query_text': 'unternehmer doppelte buchhaltung führen Bitte mit Verweisen?',
        'consensus_law': 'UGB',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v7',
        'query_text': 'angaben impressum unternehmenswebsite enthalten?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q75_impressumspflicht_website_v8',
        'query_text': 'angaben impressum unternehmenswebsite enthalten Bitte mit Verweisen?',
        'consensus_law': 'ECG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v7',
        'query_text': 'rechte habe gesellschafter gmbh einsicht unterlagen gesellschaft?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q76_gmbh_einsicht_rechnungslegung_v8',
        'query_text': 'rechte habe gesellschafter gmbh einsicht unterlagen gesellschaft Bitte mit Verweisen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v7',
        'query_text': 'läuft rechtliche gründung gmbh?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q77_gmbh_gruendung_ablauf_v8',
        'query_text': 'läuft rechtliche gründung gmbh Bitte mit Verweisen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v7',
        'query_text': 'wichtige entscheidungen aktiengesellschaft hauptversammlung erforderlich?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q78_hauptversammlung_beschluesse_v8',
        'query_text': 'wichtige entscheidungen aktiengesellschaft hauptversammlung erforderlich Bitte mit Verweisen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v7',
        'query_text': 'pflichten vorstand aktiengesellschaft gegenüber aktionären?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q79_pflichten_vorstand_aktiengesellschaft_v8',
        'query_text': 'pflichten vorstand aktiengesellschaft gegenüber aktionären Bitte mit Verweisen?',
        'consensus_law': 'AktG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v7',
        'query_text': 'haftung trifft geschäftsführer gmbh pflichtverletzungen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q80_haftung_geschaeftsfuehrer_gmbh_v8',
        'query_text': 'haftung trifft geschäftsführer gmbh pflichtverletzungen Bitte mit Verweisen?',
        'consensus_law': 'GmbHG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v7',
        'query_text': 'läuft verwaltungsverfahren behörde grundsätzlich gesetzliche Bestimmungen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q81_ablauf_verwaltungsverfahren_v8',
        'query_text': 'läuft verwaltungsverfahren behörde grundsätzlich?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v7',
        'query_text': 'rechte habe baugenehmigung verweigert?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q82_baugenehmigung_verwehrt_rechte_v8',
        'query_text': 'rechte habe baugenehmigung verweigert Bitte mit Verweisen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v7',
        'query_text': 'bescheid magistrats beschwerde beim verwaltungsgericht einlegen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q83_beschwerde_verwaltungsgericht_v8',
        'query_text': 'bescheid magistrats beschwerde beim verwaltungsgericht einlegen Bitte mit Verweisen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v7',
        'query_text': 'fristen gelten beschwerde verwaltungsbescheid?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q84_beschwerdefrist_verwaltungsbescheid_v8',
        'query_text': 'fristen gelten beschwerde verwaltungsbescheid Bitte mit Verweisen?',
        'consensus_law': 'VwGVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v7',
        'query_text': 'gewährt akteneinsicht laufenden verwaltungsverfahren?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q85_akteneinsicht_verwaltungsverfahren_v8',
        'query_text': 'gewährt akteneinsicht laufenden verwaltungsverfahren Bitte mit Verweisen?',
        'consensus_law': 'AVG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v7',
        'query_text': 'tun wegen meines geschlechts arbeit benachteiligt werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q86_diskriminierung_geschlecht_arbeit_v8',
        'query_text': 'tun wegen meines geschlechts arbeit benachteiligt werde Bitte mit Verweisen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v7',
        'query_text': 'arbeitgeber wegen meiner ethnischen herkunft einstellen gesetzliche Bestimmungen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q87_diskriminierung_ethnische_herkunft_v8',
        'query_text': 'arbeitgeber wegen meiner ethnischen herkunft einstellen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v7',
        'query_text': 'rechte habe wegen meines alters job diskriminiert werde?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q88_diskriminierung_alter_job_v8',
        'query_text': 'rechte habe wegen meines alters job diskriminiert werde Bitte mit Verweisen?',
        'consensus_law': 'GlBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v7',
        'query_text': 'besonderen schutzbestimmungen gelten jugendliche jahren arbeit?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q89_jugendliche_arbeitsschutz_v8',
        'query_text': 'besonderen schutzbestimmungen gelten jugendliche jahren arbeit Bitte mit Verweisen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v7',
        'query_text': 'viele stunden 15-jähriger ferien arbeiten?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q90_ferialjob_arbeitszeit_15_v8',
        'query_text': 'viele stunden 15-jähriger ferien arbeiten Bitte mit Verweisen?',
        'consensus_law': 'KJBG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v7',
        'query_text': 'schutzfristen gelten geburt werdende mütter?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q91_mutterschutz_schutzfristen_v8',
        'query_text': 'schutzfristen gelten geburt werdende mütter Bitte mit Verweisen?',
        'consensus_law': 'MSchG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v7',
        'query_text': 'elternkarenz geringfügig arbeiten?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q92_karenz_beschaeftigung_v8',
        'query_text': 'elternkarenz geringfügig arbeiten Bitte mit Verweisen?',
        'consensus_law': 'VKG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v7',
        'query_text': 'lange kinderbetreuungsgeld beziehen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q93_kinderbetreuungsgeld_dauer_v8',
        'query_text': 'lange kinderbetreuungsgeld beziehen Bitte mit Verweisen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v7',
        'query_text': 'modelle kinderbetreuungsgeldes gibt wähle?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q94_kinderbetreuungsgeld_modelle_v8',
        'query_text': 'modelle kinderbetreuungsgeldes gibt wähle Bitte mit Verweisen?',
        'consensus_law': 'KBGG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v7',
        'query_text': 'stelle wenden oberösterreich sozialhilfe beantragen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q95_sozialhilfe_antrag_stelle_v8',
        'query_text': 'stelle wenden oberösterreich sozialhilfe beantragen Bitte mit Verweisen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v7',
        'query_text': 'mein vermögen berechnung sozialhilfe berücksichtigt?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q96_sozialhilfe_vermoegen_anrechnung_v8',
        'query_text': 'mein vermögen berechnung sozialhilfe berücksichtigt Bitte mit Verweisen?',
        'consensus_law': 'SH-GG',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v7',
        'query_text': 'mache wegen diebstahls strafbar supermarkt etwas mitnehme?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q97_diebstahl_supermarkt_v8',
        'query_text': 'mache wegen diebstahls strafbar supermarkt etwas mitnehme Bitte mit Verweisen?',
        'consensus_law': 'StGB',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v7',
        'query_text': 'möglichkeiten habe strafzettel falschparken bekommen habe?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q98_strafzettel_falschparken_v8',
        'query_text': 'möglichkeiten habe strafzettel falschparken bekommen habe Bitte mit Verweisen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v7',
        'query_text': 'einspruch organstrafverfügung polizei erheben gesetzliche Bestimmungen?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q99_einspruch_organstrafverfuegung_v8',
        'query_text': 'einspruch organstrafverfügung polizei erheben?',
        'consensus_law': 'VStG',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v7',
        'query_text': 'regeln gelten benutzung e-scootern straßenverkehr?',
        'consensus_law': 'StVO',
    },
    {
        'query_id': 'Q100_regeln_e_scooter_strassenverkehr_v8',
        'query_text': 'regeln gelten benutzung e-scootern straßenverkehr Bitte mit Verweisen?',
        'consensus_law': 'StVO',
    },
]
