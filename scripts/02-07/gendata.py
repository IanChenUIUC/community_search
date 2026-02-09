import subprocess
import sys

networks = [
    "academia_edu",
    "advogato",
    "anybeat",
    "at_migrations",
    "berkstan_web",
    "bible_nouns",
    "bitcoin_alpha",
    "bitcoin_trust",
    "chess",
    "chicago_road",
    "citeseer",
    "collins_yeast",
    "cora",
    "dblp_cite",
    "dblp_coauthor_snap",
    "dnc",
    "douban",
    "drosophila_flybi",
    "elec",
    "email_enron",
    "email_eu",
    "epinions_trust",
    "faa_routes",
    "facebook_wall",
    "fediverse",
    "fly_hemibrain",
    "fly_larva",
    "foldoc",
    "google",
    "google_plus",
    "google_web",
    "inploid",
    "interactome_figeys",
    "interactome_stelzl",
    "interactome_vidal",
    "internet_as",
    "jdk",
    "jung",
    "lastfm_aminer",
    "libimseti",
    "linux",
    "livemocha",
    "lkml_reply",
    "marker_cafe",
    "marvel_universe",
    "myspace_aminer",
    "netscience",
    "new_zealand_collab",
    "notre_dame_web",
    "openflights",
    "petster",
    "pgp_strong",
    "polblogs",
    "power",
    "prosper",
    "python_dependency",
    "reactome",
    "slashdot_threads",
    "slashdot_zoo",
    "sp_infectious",
    "stanford_web",
    "topology",
    "twitter",
    "twitter_15m",
    "uni_email",
    "us_air_traffic",
    "wiki_link_dyn",
    "wiki_rfa",
    "wiki_users",
    "wikiconflict",
    "word_assoc",
    "wordnet",
    "yahoo_ads",
]


def main():
    for f in networks:
        input = f"/u/ianchen3/ianchen3/journal-sbm/datasets/empirical/{f}/network.tsv"
        output = f"/u/ianchen3/ianchen3/csearch/data/normalized/{f}/network.tsv"

        process = subprocess.run(
            ["python3", "./scripts/normalize_edgelist.py", input, output]
        )

        if process.returncode != 0:
            exit(process.returncode)


if __name__ == "__main__":
    main()
