"""Multi-turn live checks against real assistant logic, using isolated temporary CSVs."""
import csv
import json
import tempfile
from pathlib import Path
from filelock import FileLock
from assistant import Assistant, ROOT, now
from booking_state import asked_fields, arabic_script, forbidden_words, resolve_date
from storage import Storage, safe_cell

SCENARIOS = [
    {'id':'a','language':'az','turns':['badde e7joz tawle la 3 bokra 3al 9','Karim','70000001','khames ashkhas','terrace','eh tamam'],
     'expected':{'name':'Karim','phone':'70000001','party_size':5,'time':'21:00','seating':'terrace'},'day':'tomorrow'},
    {'id':'b','language':'ar','turns':['بدي احجز ٢ يوم السبت الساعة ٨ عالتراس','سامي','70000002','لا خليهن ٤','أكيد'],
     'expected':{'name':'سامي','phone':'70000002','party_size':4,'time':'20:00','seating':'terrace'},'day':'saturday'},
    {'id':'c','language':'en','turns':['table for 2 tomorrow at 8pm inside','Maya','+96171111111','actually make it terrace','yes please'],
     'expected':{'name':'Maya','phone':'+96171111111','party_size':2,'time':'20:00','seating':'terrace'},'day':'tomorrow'},
]


def reserve_conversation_run():
    day=now().date().isoformat()
    path=ROOT/'tests/conversation_eval_runs.json'
    with FileLock(str(ROOT/'data/conversation_eval.lock')):
        days=json.loads(path.read_text('utf-8')) if path.exists() else []
        if day in days:
            raise ValueError('Conversation evaluation already ran today; no API calls made.')
        days.append(day)
        path.write_text(json.dumps(days,indent=2)+'\n','utf-8')


async def run_case(provider, scenario):
    transcript=[]
    errors=[]
    counts={}
    expected={**scenario['expected'],'date':resolve_date(scenario['day'],now())}
    with tempfile.TemporaryDirectory(prefix='restaurant-conversation-') as directory:
        bot=Assistant(provider,Storage(directory,local_only=True),dry_run=False)
        csv_path=Path(directory)/'bookings.csv'
        for index,user in enumerate(scenario['turns']):
            try:
                actual=await bot.reply(user)
            except Exception as exc:
                actual='Conversation turn failed: '+type(exc).__name__
                errors.append(f'Turn {index+1}: {type(exc).__name__}')
            transcript.append({'user':user,'assistant':actual,'language':bot.language,'draft':dict(bot.booking_draft)})
            if bot.language != scenario['language']:
                errors.append(f'Turn {index+1}: session language changed to {bot.language}.')
            if scenario['language']=='az' and arabic_script(actual):
                errors.append(f'Turn {index+1}: Arabic script in Arabizi response.')
            if scenario['language']=='ar' and forbidden_words(actual):
                errors.append(f'Turn {index+1}: forbidden non-Lebanese words.')
            for field in asked_fields(actual):
                counts[field]=counts.get(field,0)+1
                if field in bot.booking_draft and field not in bot.draft_errors:
                    errors.append(f'Turn {index+1}: re-asked collected {field}.')
            if index < len(scenario['turns'])-1 and csv_path.exists():
                errors.append(f'Turn {index+1}: saved before final confirmation.')
        for field in (['name','phone'] if scenario['id']!='c' else list(counts)):
            if counts.get(field,0)>1:
                errors.append(f'Asked for {field} more than once.')
        booking=bot.last_booking or {}
        for field,value in expected.items():
            if booking.get(field)!=value:
                errors.append(f'Final {field}: expected {value!r}, got {booking.get(field)!r}.')
        rows=[]
        if csv_path.exists():
            with csv_path.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
        if len(rows)!=1:
            errors.append(f'Expected one saved CSV booking, got {len(rows)}.')
        elif any(rows[0].get(field)!=safe_cell(value) for field,value in expected.items()):
            errors.append('Saved CSV fields do not match the expected correction.')
        if not any(t['name']=='save_booking' and t['result'].get('ok') and not t['result'].get('dry_run') for t in bot.captured_tools):
            errors.append('No successful committed save_booking tool result.')
        return {'id':scenario['id'],'language':scenario['language'],'verdict':'FAIL' if errors else 'PASS',
                'errors':errors,'expected':expected,'booking':booking,'saved_csv_rows':len(rows),
                'transcript':transcript,'tool_calls':bot.captured_tools}


def write_conversation_report(results,provider,started):
    passed=sum(r['verdict']=='PASS' for r in results)
    payload={'started':started,'passed':passed,'total':3,'usage':provider.usage,'storage':'isolated temporary CSV; no customer data/Sheets writes','results':results}
    (ROOT/'tests/conversation_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf-8')
    lines=[f'# Conversation results\n\n{passed}/3 passed; {len(results)}/3 completed.\n\nRun: {started}\n\nDeterministic assertions; no LLM judge. Saved to isolated temporary CSV files.\n']
    for model,usage in provider.usage.items():
        lines.append(f"{model}: {usage['prompt_tokens']} input + {usage['completion_tokens']} output = {usage['total_tokens']} tokens.\n")
    for result in results:
        lines.append(f"\n## Conversation {result['id']} — {result['verdict']}\n")
        for turn in result['transcript']:
            lines.append('Customer: '+turn['user']+'\n\nAssistant:\n'+turn['assistant']+'\n')
        lines.append('Final booking: '+json.dumps(result['booking'],ensure_ascii=False)+'\n')
        lines.extend('FAIL: '+error+'\n' for error in result['errors'])
    (ROOT/'tests/conversation_results.md').write_text('\n'.join(lines),'utf-8')
    return passed


async def run_conversations(provider):
    started=now().isoformat()
    results=[]
    try:
        for scenario in SCENARIOS:
            result=await run_case(provider,scenario)
            results.append(result)
            write_conversation_report(results,provider,started)
            print(f"Conversation {scenario['id']}: {result['verdict']}",flush=True)
            for turn in result['transcript']:
                print('CUSTOMER: '+turn['user']+'\nASSISTANT: '+turn['assistant'],flush=True)
            for error in result['errors']:
                print('FAIL: '+error,flush=True)
    finally:
        passed=write_conversation_report(results,provider,started)
        tokens=sum(u['total_tokens'] for u in provider.usage.values())
        print(f'Conversations: {passed}/3 passed. API tokens: {tokens:,}.',flush=True)
        await provider.client.close()
