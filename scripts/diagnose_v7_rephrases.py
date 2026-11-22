"""Supplemental wording-transfer test with frozen, hand-written public question paraphrases."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_pairs_v7 as primary
from mp3_sql.common import digest, read_jsonl, write_json, write_jsonl

WORK = ROOT / "runs/v7_rephrases"
PRIMARY = primary.PUBLIC
PUBLIC = WORK / "public.jsonl"
ADAPTERS = {
    "baseline": ROOT / "artifacts/adapters/onpolicy_v001/epoch_02",
    "ordinary": ROOT / "artifacts/adapters/v7_ordinary/epoch_01",
    "paired": ROOT / "artifacts/adapters/v7_paired/epoch_01",
}

# Each entry maps an original public intent to a paraphrase. Neither side labels
# nor draft SQL are consulted when assigning a replacement.
TEXT = {
    "How many records are in customers?": "What is the customer headcount?",
    "How many different customer address id values are recorded in customers?": "Across the customer list, how many addresses appear, with a shared address contributing only once?",
    "Count all entries in publication, including repeated book id values.": "What is the total number of publication records on file?",
    "Count the distinct book represented by book id in publication.": "How many individual books appear in the publication register, even if a book was published several times?",
    "What is the total number of rows recorded in inventory?": "How many inventory items are on file altogether?",
    "In inventory, how many separate film id values occur, counting repeated values just once?": "Ignoring additional copies of a film, how many films does the inventory contain?",
    "How many records are in lots?": "What is the total number of lots on the register?",
    "How many different investor id values are recorded in lots?": "How many investors hold one or more of the recorded lots?",
    "Count all entries in player, including repeated school id values.": "How large is the player roster?",
    "Count the distinct school represented by school id in player.": "How many schools have at least one player on the roster?",
    "What is the total number of rows recorded in order items?": "How many individual order lines have been entered?",
    "In order items, how many separate order id values occur, counting repeated values just once?": "How many orders have at least one recorded line item?",
    "How many records are in photos?": "What is the photograph collection's size?",
    "How many different mountain id values are recorded in photos?": "How many mountains are represented in the photograph collection?",
    "Count all entries in loan, including repeated cust id values.": "How many loans have been entered in the loan ledger?",
    "Count the distinct customer represented by cust id in loan.": "How many people in the customer register have one or more loans on file?",
    "How many mountain records are there in total, regardless of entries in photos?": "How many mountains are listed, photographed or otherwise?",
    "How many matching pairs of mountain and photos records are linked by photos.mountain id?": "For every photo attached to a listed mountain, add one to a tally. What is the tally?",
    "Count each record in customers once, including records with no related lessons.": "What is the complete customer headcount, whether anyone has booked lessons or not?",
    "Count every matching customers and lessons combination connected through customer id, counting each match.": "What is the total number of lesson bookings that can be attributed to a listed customer?",
    "What is the size of the book table, including any records with no matching publication entry?": "How many books does the catalog list, published or unpublished?",
    "How many times do records from book match an entry in publication via book id, counting all occurrences?": "What is the total number of publications attributable to catalogued books, including multiple publications of the same book?",
    "How many department records are there in total, regardless of entries in course?": "How many departments are on file, including departments offering no courses?",
    "How many matching pairs of department and course records are linked by course.dno?": "What is the total tally of courses that belong to a registered department?",
    "Count each record in customer once, including records with no related loan.": "How many customers are on the books, borrowers and nonborrowers together?",
    "Count every matching customer and loan combination connected through cust id, counting each match.": "How many loans can be attributed to an existing customer, with every loan included?",
    "What is the size of the drama workshop groups table, including any records with no matching services entry?": "How many drama workshop groups are listed, even if some offer no services?",
    "How many times do records from drama workshop groups match an entry in services via workshop group id, counting all occurrences?": "Add up the services attached to registered drama workshop groups. What is the total number of such service records?",
    "How many investors records are there in total, regardless of entries in lots?": "What is the investor headcount, including investors without holdings?",
    "How many matching pairs of investors and lots records are linked by lots.investor id?": "What is the number of lots attributed to registered investors, with each lot contributing once?",
    "Count each record in school once, including records with no related player.": "How many schools appear on the register, including those without players?",
    "Count every matching school and player combination connected through school id, counting each match.": "Across the registered schools, what is the total number of player assignments?",
    "How many records in phone have price greater than 40?": "Of the phones on file, how many cost more than 40?",
    "How many records in phone have price at least 40?": "How many phones cost no less than 40?",
    "Count mountain entries with height strictly above 50.": "How many listed mountains exceed 50 in height?",
    "Count mountain entries with height 50 or above.": "How many listed mountains reach a height of 50 or more?",
    "Among the film records, how many have length below 50, excluding 50 itself?": "How many films are shorter than 50?",
    "Among the film records, how many have length 50 or below, including equality?": "Of the films on file, how many are no longer than 50?",
    "How many records in campuses have year less than 20?": "How many campus entries have a recorded year earlier than 20?",
    "How many records in campuses have year at most 20?": "How many campus entries have a recorded year no later than 20?",
    "Count invoices entries with product id strictly above 15.": "How many invoices have a product ID exceeding 15?",
    "Count invoices entries with product id 15 or above.": "How many invoices have a product ID no smaller than 15?",
    "Among the department records, how many have num employees above 10, excluding 10 itself?": "How many departments employ more than ten people?",
    "Among the department records, how many have num employees 10 or above, including equality?": "How many departments have ten or more employees?",
    "How many records in school have enrollment less than 20?": "How many schools have fewer than twenty students enrolled?",
    "How many records in school have enrollment at most 20?": "How many schools have no more than twenty students enrolled?",
    "Count book entries with issues strictly below 100.": "How many books have fewer than one hundred issues?",
    "Count book entries with issues 100 or below.": "How many books have no more than one hundred issues?",
}


def prepare():
    rows = read_jsonl(PRIMARY)
    if {r["question"] for r in rows} != set(TEXT) or len(set(TEXT.values())) != len(rows) or len(rows) != 48:
        raise ValueError("Reformulations must cover precisely all 48 public cases")
    previous = set(TEXT)
    for arm in ("ordinary", "paired"):
        for r in read_jsonl(primary.DATA / arm / "train.jsonl"):
            content = r["messages"][1]["content"]
            payload = json.loads(content.split("TASK_DATA\n", 1)[1].split("\nEND_TASK_DATA", 1)[0])
            previous.add(payload["question"])
    if previous & set(TEXT.values()):
        raise ValueError("A reformulation already appears in the original or fit set")
    if WORK.exists():
        raise FileExistsError(WORK)
    WORK.mkdir(parents=True)
    rewritten = [{**r, "question":TEXT[r["question"]]} for r in rows]
    if any({k:v for k,v in o.items() if k!="question"} != {k:v for k,v in r.items() if k!="question"}
           for o,r in zip(rows,rewritten)):
        raise RuntimeError("Modified non-wording input")
    write_jsonl(PUBLIC, rewritten)
    write_json(WORK / "frozen.json", {
        "frozen_utc":datetime.now(timezone.utc).isoformat(), "script_sha256":digest(Path(__file__)),
        "original_public_sha256":digest(PRIMARY), "rephrased_public_sha256":digest(PUBLIC),
        "private_sha256":digest(primary.PRIVATE),"fit_questions_crosschecked":len(previous),
        "pairs":len({r["pair_id"] for r in rows}), "tasks":len(rows),
        "method":"One supplementary English paraphrase for every member of the prespecified heldout population; only question text changed. No selection on predictions. Labels, schemas, initial SQL, support results and witnesses retained.",
    })


def verify():
    frozen=json.loads((WORK/"frozen.json").read_text())
    assert frozen["rephrased_public_sha256"]==digest(PUBLIC)
    assert frozen["original_public_sha256"]==digest(PRIMARY)
    assert frozen["private_sha256"]==digest(primary.PRIVATE)
    assert frozen["script_sha256"]==digest(Path(__file__))
    return frozen


def summary():
    frozen=verify()
    comparisons={}
    for arm in primary.ARMS:
        labels=json.loads((WORK/arm/"score.json").read_text())
        original=json.loads((ROOT/"runs/v7_pair_eval"/arm/"score.json").read_text())
        assert original["run"]["adapter_sha256"]==labels["run"]["adapter_sha256"]
        assert original["private_sha256"]==labels["private_sha256"]==frozen["private_sha256"]
        assert original["witness_fixtures"]==labels["witness_fixtures"]
        assert original["run"]["public_fixtures"]==labels["run"]["public_fixtures"]
        assert original["run"]["device"]==labels["run"]["device"]
        assert original["run"]["config_sha256"]==labels["run"]["config_sha256"]
        def successes(items):
            counts=Counter(m["pair_id"] for m in items if m["correct"])
            return {p for p,n in counts.items() if n==2}
        before,after=successes(original["members"]),successes(labels["members"])
        comparisons[arm]={"primary":original["overall"], "paraphrased":labels["overall"],
                          "paraphrased_by_category":labels["by_category"], "gained_both":len(after-before),
                          "lost_both":len(before-after),"same_both":len(before&after)}
    write_json(ROOT/"reports/v7_rephrases.json",{"population":frozen, "arms":comparisons})
    print({a:(s["primary"]["both"],s["paraphrased"]["both"]) for a,s in comparisons.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","predict","grade","summary"])
    parser.add_argument("--arm",choices=primary.ARMS)
    arg=parser.parse_args()
    if arg.stage=="prepare":
        prepare()
        return
    verify()
    if arg.stage=="summary":
        summary()
        return
    if not arg.arm:
        parser.error("Specifying --arm is required for inference and grading")
    # These process-local overrides leave the original experiment and its CLI untouched.
    primary.PUBLIC=PUBLIC
    primary.WORK=WORK
    if arg.stage=="predict":
        primary.predict(arg.arm,ADAPTERS[arg.arm],"cuda")
    else:
        primary.grade(arg.arm)


if __name__=="__main__":
    main()
