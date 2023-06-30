import unittest

from rewards import bleu_reward_estimation, compute_sentence_bleu, correct_ques_num_reward_estimation


class TestEvaluation(unittest.TestCase):

    def test_bleu_reward(self):
        reference = ["Trying to improve education.\n Ideally, by adding 1 + 1 is equal to 3."]
        prediction = ["Trying to improve education.\n Ideally, by adding 1 + 1 is equal to 3."]
        bleu = compute_sentence_bleu(reference[0], prediction[0])
        self.assertEqual(bleu, 1.0)
        self.assertListEqual(bleu_reward_estimation(reference, prediction, 1), [1.0])

        reference = ["city area education processing learning"]
        prediction = ["city area education processing"]
        bleu = compute_sentence_bleu(reference[0], prediction[0])
        self.assertGreaterEqual(bleu, 0.7)
        self.assertGreaterEqual(bleu_reward_estimation(reference, prediction, 1)[0], 0.7)

    def test_question_count(self):
        reference = ["How many questions should we ask?\n Is it enough?"]
        prediction = ["How many questions should we ask?\n Is it enough?"]
        self.assertListEqual([1.0], correct_ques_num_reward_estimation(reference, prediction, 1))
        self.assertListEqual([1.0, 1.0], correct_ques_num_reward_estimation(reference, prediction * 2, 2))

        prediction = ["How many questions should we ask?\n Is it enough?\n Is more better?"]
        self.assertListEqual([0.5], correct_ques_num_reward_estimation(reference, prediction, 1))

        prediction = ["How many questions should we ask?\n Is it enough?\n Is more better?\n And more?"]
        self.assertListEqual([0.0], correct_ques_num_reward_estimation(reference, prediction, 1))

        prediction = ["Is less enough?"]
        self.assertListEqual([0.5], correct_ques_num_reward_estimation(reference, prediction, 1))
