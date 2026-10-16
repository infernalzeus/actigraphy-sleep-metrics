import unittest
import hashlib
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ck_input import counts_to_minutes, attach_ck_input
from sri import cole_kripke, compute_SRI, score_sleep_wake, _ggir_scores


def minute_frame(n=20):
    return pd.DataFrame({'Time':pd.date_range('2026-07-14 13:57:47',periods=n,freq='60s'),
                         'SVM':np.zeros(n),'CK_activity':np.zeros(n)})


class CKTests(unittest.TestCase):
    def test_original_ggir_r_fixtures(self):
        directory=Path(__file__).parent/'fixtures'
        inputs=pd.read_csv(directory/'ggir_counts.csv')
        outputs=pd.read_csv(directory/'ggir_states.csv')
        for name, group in inputs.groupby('case',sort=False):
            with self.subTest(case=name):
                frame=pd.DataFrame({'Time':pd.date_range('2026-07-14 13:57:47',periods=len(group),freq='2s'),
                                    'CK_count':group['count'].to_numpy()})
                expected=outputs[outputs['case']==name]
                states=score_sleep_wake(frame)
                np.testing.assert_array_equal(states,expected.wake)
                activity=counts_to_minutes(frame)
                # Verify weighted scores as well as binary labels, including borders.
                np.testing.assert_allclose(_ggir_scores(activity.to_numpy(),.001),expected.score,atol=1e-12)
                np.testing.assert_array_equal(cole_kripke(pd.DataFrame({'Time':activity.index,'CK_activity':activity.to_numpy()})),expected.wake)

    def test_threshold_and_prefix(self):
        for value, expected in [(0,0),(1.51,1),(1.5,0)]:
            d=minute_frame();d.CK_activity=value
            states=cole_kripke(d)
            np.testing.assert_array_equal(states[:4],1)
            np.testing.assert_array_equal(states[4:],expected)
        # Exact threshold: a zero signal with threshold=0 is wake everywhere.
        np.testing.assert_array_equal(cole_kripke(minute_frame(),threshold=0),1)

    def test_missing_counts_follow_g_sib_det(self):
        d=minute_frame();d.loc[8,'CK_activity']=np.nan
        expected=cole_kripke(d)
        result=score_sleep_wake(d.drop(index=8).sample(frac=1,random_state=3))
        np.testing.assert_equal(result.to_numpy(),expected)
        self.assertFalse(np.isnan(expected).any())
        np.testing.assert_equal(cole_kripke(pd.concat([d,d.iloc[[0]]])),np.r_[expected,expected[0]])
        conflict=pd.concat([d,d.iloc[[0]].assign(CK_activity=5)])
        with self.assertRaises(ValueError): cole_kripke(conflict)

    def test_reduction_and_timestamp_phase(self):
        times=pd.date_range('2026-07-14 13:57:47',periods=91,freq='2s')
        d=pd.DataFrame({'Time':times,'CK_count':np.full(91,4.)})
        out=counts_to_minutes(d)
        np.testing.assert_equal(out.to_numpy(),[20,20,20,np.nan])
        self.assertEqual(out.index[0],times[0])
        broken=counts_to_minutes(d.drop(index=35))
        self.assertAlmostEqual(broken.iloc[1],29*4/6)
        wrong=d.copy();wrong.loc[10,'Time']+=pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError): counts_to_minutes(wrong)
        wrong=d.copy();wrong.loc[10,'CK_count']=-1
        with self.assertRaises(ValueError): counts_to_minutes(wrong)

    def test_sri_schema_and_ggir_missing_policy(self):
        d=minute_frame(2880)
        expected_keys={'SRI','n_days','n_valid_epoch_pairs','pct_epochs_sleep','epoch_sec',
                       'classifier','ck_scale','ck_threshold','activity_normalization','states_matrix','dates'}
        result=compute_SRI(d)
        self.assertEqual(set(result),expected_keys)
        self.assertAlmostEqual(result['SRI'],200*(1440-4)/1440-100)
        self.assertEqual(result['n_valid_epoch_pairs'],1440)
        self.assertEqual(result['states_matrix'].shape,(3,1440))
        d.CK_activity=np.nan
        np.testing.assert_array_equal(compute_SRI(d)['states_matrix'],result['states_matrix'])
        with self.assertRaises(ValueError): compute_SRI(d,epoch_sec=30)
        self.assertEqual(compute_SRI(d.iloc[:0])['n_days'],0)

    def test_sidecar_validation_and_old_csv(self):
        d=minute_frame(20).drop(columns='CK_activity')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'sample.csv';d.to_csv(path,index=False)
            pd.testing.assert_frame_equal(attach_ck_input(d,path),d)
            counts=Path(td)/'sample_2s.csv'
            pd.DataFrame({'Time':pd.date_range(d.Time[0],periods=600,freq='2s'),
                          'CK_count':np.ones(600)}).to_csv(counts,index=False)
            info={'schema_version':1,'epoch_sec':2,'axis':'y','file':counts.name,
                  'main_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'counts_csv_sha256':hashlib.sha256(counts.read_bytes()).hexdigest()}
            path.with_suffix('.metadata.json').write_text(json.dumps({'preprocessing':{'ck_counts':info}}))
            loaded=attach_ck_input(d,path)
            np.testing.assert_array_equal(loaded.CK_activity.to_numpy(),5)
            np.testing.assert_array_equal(loaded.SVM.to_numpy(),d.SVM.to_numpy())
            counts.write_text('corrupted')
            with self.assertRaises(ValueError): attach_ck_input(d,path)

    def test_short_array_legacy_does_not_crash(self):
        for n in range(8): self.assertEqual(len(cole_kripke(np.zeros(n))),n)

if __name__=='__main__': unittest.main()
