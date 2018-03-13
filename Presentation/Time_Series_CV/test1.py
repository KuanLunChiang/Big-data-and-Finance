import unittest


class Test_test1(unittest.TestCase):
    
    def test_A(self):
        from Time_Series.Time_Horizon import Time_Horizon
        a = Time_Horizon(1)
        b = a.add(1,2)
        self.assertEqual(b,3)


    def test_b (self):
        a = Time_Horizon()
        b = a.add(1,1)
        print(b)
        self.assertTrue(True)




if __name__ == '__main__':
    unittest.main()
