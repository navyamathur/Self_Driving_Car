import socket
s = socket.socket()         
print ("Socket successfully created")
port = 12346              
s.bind(('', port))        
print ("socket binded to %s" %(port))
s.listen(5)     
print ("socket is listening")           
c, addr = s.accept()     
print ('Got connection from', addr)
c.sendall('Thank you for connecting'.encode('utf-8'))
  
while True:
   data=float(c.recv(1024))
   print(data)
   #com=input("Enter msg: ")
   #c.sendall(com.encode('utf-8'))
c.close()
