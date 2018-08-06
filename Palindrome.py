class Palindrome:
    def __init__(self):
        #nohthing here
        self = self
        
    def isPalindrome(self, word):
        return word.lower() == word.lower()[::-1]
    
p = Palindrome()
print (p.isPalindrome('poop'))
print (p.isPalindrome('shit'))
