import json
import unittest

from dataloader import sentence_planning_window, iterative_sentence_planning_with_separators, planning_strategy, \
    critic_data_preprocessing


class TestParsing(unittest.TestCase):

    def test_iterative_planning_partial(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq = sentence_planning_window(data)
        expected_input_seq = ['Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4.', " She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10.", " She also receives an overtime pay of 1.2 times her regular hourly rate.", " hours this week, how much are her earnings for this week?"]
        expected_output_seq = ['How many eggs does Janet sell?', "How much does Janet make at the farmers' market?",
                               "How many hours of overtime pay does Eliza receive?", "How much is Eliza's overtime rate?", "How much will Eliza receive in overtime pay? How much is Eliza's regular weekly earning? How much are Eliza's earnings for this week?"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)

    def test_iterative_planning_with_separators(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq, _ = iterative_sentence_planning_with_separators(data)
        expected_input_seq = ["[SEP] Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. [/SEP] She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                              "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. [SEP] She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? [/SEP]",
                              "[SEP] Eliza's rate per hour for the 1 40 hours she works each week is $10. [/SEP] She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. [SEP] She also receives an overtime pay of 1.2 times her regular hourly rate. [/SEP] If Eliza worked for 45 hours this week, how much are her earnings for this week?",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. [SEP] If Eliza worked for 45 hours this week, how much are her earnings for this week? [/SEP]"]
        expected_output_seq = ['How many eggs does Janet sell?', "How much does Janet make at the farmers' market?",
                               "How many hours of overtime pay does Eliza receive?",
                               "How much is Eliza's overtime rate?",
                               "How much will Eliza receive in overtime pay? How much is Eliza's regular weekly earning? How much are Eliza's earnings for this week?"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)


    def test_iterative_planning_with_separators_and_operator(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq, _ = iterative_sentence_planning_with_separators(data, planning="operator")
        expected_input_seq = ["[SEP] Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. [/SEP] << - - >> She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                              "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. [SEP] She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? [/SEP] << * >>",
                              "[SEP] Eliza's rate per hour for the 1 40 hours she works each week is $10. [/SEP] << - >> She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. [SEP] She also receives an overtime pay of 1.2 times her regular hourly rate. [/SEP] << * >> If Eliza worked for 45 hours this week, how much are her earnings for this week?",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. [SEP] If Eliza worked for 45 hours this week, how much are her earnings for this week? [/SEP] << * >> << * >> << + >>"]
        expected_output_seq = ['How many eggs does Janet sell?', "How much does Janet make at the farmers' market?",
                               "How many hours of overtime pay does Eliza receive?",
                               "How much is Eliza's overtime rate?",
                               "How much will Eliza receive in overtime pay? How much is Eliza's regular weekly earning? How much are Eliza's earnings for this week?"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)

    def test_iterative_planning_with_separators_and_equation(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq, _ = iterative_sentence_planning_with_separators(data, planning="equation")
        expected_input_seq = ["[SEP] Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. [/SEP] <<16-3-4=9>> She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                              "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. [SEP] She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? [/SEP] <<9*2=18>>",
                              "[SEP] Eliza's rate per hour for the 1 40 hours she works each week is $10. [/SEP] <<45-40=5>> She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. [SEP] She also receives an overtime pay of 1.2 times her regular hourly rate. [/SEP] <<10*1.2=12>> If Eliza worked for 45 hours this week, how much are her earnings for this week?",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. [SEP] If Eliza worked for 45 hours this week, how much are her earnings for this week? [/SEP] <<12*5=60>> <<10*40=400>> <<400+60=460>>"]
        expected_output_seq = ['How many eggs does Janet sell?', "How much does Janet make at the farmers' market?",
                               "How many hours of overtime pay does Eliza receive?",
                               "How much is Eliza's overtime rate?",
                               "How much will Eliza receive in overtime pay? How much is Eliza's regular weekly earning? How much are Eliza's earnings for this week?"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)

    def test_plain_planning_with_operators(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq = planning_strategy(data, "operator")
        expected_input_seq = ["Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? [SEP] << - - >> << * >>",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week? [SEP] << - >> << * >> << * >> << * >> << + >>"]
        expected_output_seq = ["How many eggs does Janet sell? \n How much does Janet make at the farmers' market? \n",
                               "How many hours of overtime pay does Eliza receive? \n How much is Eliza's overtime rate? \n How much will Eliza receive in overtime pay? \n How much is Eliza's regular weekly earning? \n How much are Eliza's earnings for this week? \n"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)

    def test_plain_planning_with_equations(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq = planning_strategy(data, "equation")
        expected_input_seq = ["Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? [SEP] <<16-3-4=9>> <<9*2=18>>",
                              "Eliza's rate per hour for the 1 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week? [SEP] <<45-40=5>> <<10*1.2=12>> <<12*5=60>> <<10*40=400>> <<400+60=460>>"]
        expected_output_seq = ["How many eggs does Janet sell? \n How much does Janet make at the farmers' market? \n",
                               "How many hours of overtime pay does Eliza receive? \n How much is Eliza's overtime rate? \n How much will Eliza receive in overtime pay? \n How much is Eliza's regular weekly earning? \n How much are Eliza's earnings for this week? \n"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)

    def test_critic_operator_data_preprocessing(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq = critic_data_preprocessing(data, "operator")
        expected_input_seq = [
            "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "Eliza's rate per hour for the 1 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?"]
        expected_output_seq = ["<< - - >> << * >>",
                               "<< - >> << * >> << * >> << * >> << + >>"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)

    def test_critic_equation_data_preprocessing(self):
        with open("./data/data_subset.jsonl") as f:
            data = [json.loads(line) for line in f.readlines() if line]
        input_seq, output_seq = critic_data_preprocessing(data, "equation")
        expected_input_seq = [
            "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "Eliza's rate per hour for the 1 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?"]
        expected_output_seq = ["<<16-3-4=9>> <<9*2=18>>",
                               "<<45-40=5>> <<10*1.2=12>> <<12*5=60>> <<10*40=400>> <<400+60=460>>"]
        self.assertListEqual(expected_input_seq, input_seq)
        self.assertListEqual(expected_output_seq, output_seq)
