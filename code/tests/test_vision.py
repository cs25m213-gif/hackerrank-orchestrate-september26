import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import base64
from argparse import Namespace
import csv
from decimal import Decimal
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from common import COLUMNS
from vision import VisionExtractor, Usage, VisionError, ReviewRequired, validate_fact, cost_for, DEFAULT_MODEL
from estimation import estimate_expense, first_safe_offset, horizon_quantile
from engine import feasible
from datetime import date, timedelta
from main import run

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jvXkAAAAASUVORK5CYII=')
FACT = dict(amount='33.50',currency='USD',confidence='high',source_quote='Total $33.50',reason='Fare after change.')
EVENT = dict(event_id='new_event',user_id='new_user',description='Taxi fare',category='transport',currency='USD',
             direction='debit',amount='',event_date='2026-01-02',settlement_date='2026-01-02',status='pending',
             event_type='expense',linked_event_id='',flexibility='fixed',minimum_allowed_amount='')

def response(fact=FACT, **overrides):
    return dict(status='completed',model=DEFAULT_MODEL,
                output=[{'type':'message','content':[{'type':'output_text','text':json.dumps(fact)}]}],
                usage={'input_tokens':1000,'output_tokens':100,'input_tokens_details':{'cached_tokens':200}}, **overrides)

class VisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.png = self.path / 'new.png';self.png.write_bytes(PNG)
        self.key = patch.dict(os.environ, {'OPENAI_API_KEY':'fake-test-key'})
        self.key.start();self.addCleanup(self.key.stop)

    def extractor(self, transport, **kwargs):
        return VisionExtractor(self.path / 'cache', transport=transport, sleeper=lambda _:None, **kwargs)

    def test_new_image_uses_api_then_reuses_cache(self):
        calls=[]
        def transport(payload,key,timeout):
            calls.append(payload)
            self.assertFalse(payload['store'])
            self.assertTrue(payload['text']['format']['strict'])
            self.assertEqual(payload['input'][0]['content'][1]['detail'],'high')
            return response()
        ex=self.extractor(transport)
        self.assertEqual(ex.extract(self.png,EVENT)['amount'],'33.50')
        self.assertEqual(ex.extract(self.png,EVENT)['amount'],'33.50')
        self.assertEqual(len(calls),1)
        self.assertEqual(ex.usage.cache_hits,1)
        self.assertEqual(ex.usage.calls[0]['input_tokens'],1000)

    def test_same_image_different_event_context_is_not_reused(self):
        calls=[]
        def transport(*args):calls.append(1);return response()
        ex=self.extractor(transport);ex.extract(self.png,EVENT)
        ex.extract(self.png,{**EVENT,'settlement_date':'2026-01-03'})
        self.assertEqual(len(calls),2)

    def test_changed_image_same_name_invalidates_cache(self):
        calls=[]
        def transport(*args):calls.append(1);return response()
        ex=self.extractor(transport);ex.extract(self.png,EVENT)
        self.png.write_bytes(PNG+b'changed')
        ex.extract(self.png,EVENT);self.assertEqual(len(calls),2)

    def test_uncertainty_escalates_only_that_image(self):
        calls=[]
        def transport(payload,*args):
            calls.append(payload['model'])
            return response({**FACT,'confidence':'medium'} if len(calls)==1 else FACT)
        ex=self.extractor(transport);ex.extract(self.png,EVENT)
        self.assertEqual(calls,[ex.model,ex.fallback])

    def test_uncertain_final_result_is_not_cached(self):
        ex=self.extractor(lambda *args:response({**FACT,'amount':None,'confidence':'needs_review'}))
        with self.assertRaises(ReviewRequired):ex.extract(self.png,EVENT)
        self.assertEqual(list((self.path/'cache').glob('*.json')),[])

    def test_injection_cannot_add_output_fields(self):
        with self.assertRaises(ValueError):validate_fact({**FACT,'amount_safe_to_pay':'999999'},EVENT)

    def test_currency_mismatch_and_negative_money_rejected(self):
        for fact in ({**FACT,'currency':'INR'},{**FACT,'amount':'-1'},{**FACT,'amount':'NaN'}):
            with self.assertRaises(ValueError):validate_fact(fact,EVENT)

    def test_missing_key_fails_without_http(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':''}):
            with self.assertRaisesRegex(VisionError,'OPENAI_API_KEY'):
                self.extractor(lambda *args:self.fail('HTTP must not run')).extract(self.png,EVENT)

    def test_offline_cache_miss_never_calls_api(self):
        with self.assertRaisesRegex(VisionError,'Offline'):
            self.extractor(lambda *args:self.fail('HTTP must not run'),offline=True).extract(self.png,EVENT)

    def test_rate_limit_retries_and_records_unknown_usage(self):
        calls=[]
        def transport(*args):
            calls.append(1)
            if len(calls)==1:raise urllib.error.HTTPError('https://api.openai.com',429,'limited',{},None)
            return response()
        ex=self.extractor(transport);ex.extract(self.png,EVENT)
        self.assertEqual(len(calls),2)
        self.assertFalse(ex.usage.calls[0]['usage_known'])

    def test_auth_failure_not_retried_or_exposed(self):
        def transport(*args):raise urllib.error.HTTPError('https://api.openai.com',401,'secret-message',{},None)
        ex=self.extractor(transport)
        with self.assertRaises(VisionError) as e:ex.extract(self.png,EVENT)
        self.assertNotIn('secret-message',str(e.exception));self.assertEqual(len(ex.usage.calls),1)

    def test_call_budget_applies_to_fallback(self):
        ex=self.extractor(lambda *args:response({**FACT,'confidence':'medium'}),max_calls=1)
        with self.assertRaisesRegex(VisionError,'budget'):ex.extract(self.png,EVENT)
        self.assertEqual(len(ex.usage.calls),1)

    def test_cost_accounts_for_cached_input(self):
        self.assertEqual(cost_for(DEFAULT_MODEL,1000,200,100),Decimal('.00050'))
        self.assertIsNone(cost_for('unknown-model',1000,200,100))

    def test_new_dataset_end_to_end_without_seed_cache(self):
        dataset=self.path/'dataset';(dataset/'media/images').mkdir(parents=True)
        (dataset/'media/images/image_new.png').write_bytes(PNG)
        def write(name, records, fields=None):
            with (dataset/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fields or list(records[0]));w.writeheader();w.writerows(records)
        req=dict(request_id='new_request',user_id='new_user',request_date='2026-01-01',requested_amount='100',
                 desired_completion_date='2026-02-01',allows_partial_payment='false')
        profile=dict(user_id='new_user',home_currency='USD',current_available_balance='300',minimum_balance_to_keep='100',
                     payment_methods_user_will_consider='full_payment',max_installment_months='',
                     expense_categories_to_protect='transport',expense_categories_user_is_willing_to_reduce='',
                     expense_categories_user_is_willing_to_stop='')
        write('requests.csv',[req]);write('financial_profiles.csv',[profile]);write('financial_events.csv',[EVENT])
        write('images.csv',[dict(image_id='image_new',user_id='new_user',request_id='new_request',related_event_id='new_event')])
        write('messages.csv',[],['message_id']);write('request_payment_options.csv',[],['request_id'])
        write('exchange_rates.csv',[],['rate_date','from_currency','to_currency','rate'])
        args=Namespace(dataset=dataset,samples=False,output=self.path/'output.csv',image_cache=self.path/'missing.json',
                       vision_cache=self.path/'api_cache',vision_model=DEFAULT_MODEL,vision_fallback='',
                       max_api_calls=2,vision='auto',allow_legacy_cache=False,evaluation_dir=self.path/'evaluation')
        with patch('vision.post_openai',return_value=response()) as http:
            run(args)
            self.assertEqual(http.call_count,1)
        with args.output.open() as f:
            row=list(csv.DictReader(f))[0]
        self.assertEqual(list(row),COLUMNS)
        self.assertEqual(row['amount_safe_to_pay'],'100')
        self.assertEqual(row['recommended_payment_method'],'full_payment')
        self.assertIn('0.00050',(args.evaluation_dir/'usage_report.md').read_text())
        args.vision='offline'
        with patch('vision.post_openai',side_effect=AssertionError('unexpected call')):
            run(args)

    def test_unresolved_image_is_flagged_for_review_without_blocking_other_requests(self):
        # A missing image file for one user must not abort the whole run: that
        # user's request is routed to manual review (no amount invented) while
        # an unrelated user's request is still computed normally.
        dataset=self.path/'dataset';(dataset/'media/images').mkdir(parents=True)
        def write(name, records, fields=None):
            with (dataset/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fields or list(records[0]));w.writeheader();w.writerows(records)
        blocked_event={**EVENT,'event_id':'blocked_event','user_id':'blocked_user'}
        ok_event=dict(event_id='ok_event',user_id='ok_user',description='Salary',category='salary',
                     currency='USD',direction='credit',amount='500',event_date='2026-01-01',
                     settlement_date='2026-01-01',status='settled',event_type='income',
                     linked_event_id='',flexibility='fixed',minimum_allowed_amount='')
        req_blocked=dict(request_id='blocked_request',user_id='blocked_user',request_date='2026-01-01',
                         requested_amount='50',desired_completion_date='2026-02-01',allows_partial_payment='false')
        req_ok=dict(request_id='ok_request',user_id='ok_user',request_date='2026-01-01',
                   requested_amount='50',desired_completion_date='2026-02-01',allows_partial_payment='false')
        profile_blocked=dict(user_id='blocked_user',home_currency='USD',current_available_balance='300',
                             minimum_balance_to_keep='100',payment_methods_user_will_consider='full_payment',
                             max_installment_months='',expense_categories_to_protect='transport',
                             expense_categories_user_is_willing_to_reduce='',expense_categories_user_is_willing_to_stop='')
        profile_ok={**profile_blocked,'user_id':'ok_user'}
        write('requests.csv',[req_blocked,req_ok])
        write('financial_profiles.csv',[profile_blocked,profile_ok])
        write('financial_events.csv',[blocked_event,ok_event])
        # No file written for image_missing.png: the link is unresolvable.
        write('images.csv',[dict(image_id='image_missing',user_id='blocked_user',request_id='blocked_request',
                                 related_event_id='blocked_event')])
        write('messages.csv',[],['message_id']);write('request_payment_options.csv',[],['request_id'])
        write('exchange_rates.csv',[],['rate_date','from_currency','to_currency','rate'])
        args=Namespace(dataset=dataset,samples=False,output=self.path/'output_unresolved.csv',
                       image_cache=self.path/'missing3.json',vision_cache=self.path/'api_cache3',
                       vision_model=DEFAULT_MODEL,vision_fallback='',max_api_calls=2,vision='offline',
                       allow_legacy_cache=False,evaluation_dir=self.path/'evaluation3')
        run(args)  # must complete, not raise, despite the unresolvable image
        with args.output.open() as f:
            by_id={r['request_id']:r for r in csv.DictReader(f)}
        self.assertEqual(by_id['blocked_request']['amount_safe_to_pay'],'0')
        self.assertEqual(by_id['blocked_request']['affordability_status'],'not_affordable')
        self.assertEqual(by_id['ok_request']['recommended_payment_method'],'full_payment')
        audits=json.loads((args.evaluation_dir/'full_audit.json').read_text())
        blocked_audit=next(a for a in audits if a['request_id']=='blocked_request')
        self.assertTrue(blocked_audit['warnings'])

    def test_amount_must_match_quoted_evidence(self):
        with self.assertRaises(ReviewRequired):
            validate_fact({**FACT,'amount':'3350'},EVENT)

    def test_indian_number_grouping(self):
        fact={**FACT,'amount':'100000','source_quote':'Balance due: 1,00,000.00'}
        self.assertEqual(validate_fact(fact,EVENT)['amount'],'100000')

class EstimationTests(unittest.TestCase):
    def test_linear_capacity_agrees_with_payment_replay(self):
        import random
        rng=random.Random(4);start=date(2026,1,1)
        for _ in range(100):
            h=[Decimal(rng.randrange(-10,100)) for _ in range(20)];total=Decimal(50)
            expected=next((i for i in range(len(h)) if feasible(h,[(start+timedelta(days=i),total)],start)),None)
            self.assertEqual(first_safe_offset(h,total),expected)

    def test_expense_estimator_does_not_use_request_labels(self):
        hist=[dict(event_date=f'2025-{i+1:02d}-01',amount='100') for i in range(8)]
        result,method=estimate_expense(hist)
        self.assertEqual(result,100);self.assertIn('rolling_history',method)

    def test_horizon_quantile_relaxes_toward_median_as_occurrences_grow(self):
        # A single upcoming instance keeps the full single-shot 80th-percentile
        # bound; a category repeating many times in the horizon must not reserve
        # that same tail-risk margin on every instance, or the compounded total
        # wildly overstates worst-case risk (the failure this calibration fixes).
        self.assertEqual(horizon_quantile(1), Decimal('.8'))
        self.assertLess(horizon_quantile(4), Decimal('.8'))
        self.assertLess(horizon_quantile(16), horizon_quantile(4))
        self.assertGreater(horizon_quantile(1000), Decimal('.5'))

    def test_estimate_expense_avoids_compounding_conservatism_for_frequent_categories(self):
        hist=[dict(event_date=f'2025-{i+1:02d}-01',amount=v) for i,v in
              zip(range(8),['40','55','42','58','44','60','46','50'])]
        single,_=estimate_expense(hist,1)
        frequent,_=estimate_expense(hist,25)
        self.assertLess(frequent,single)

if __name__=='__main__':unittest.main()
