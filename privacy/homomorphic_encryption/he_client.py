from cryptography.hazmat.primitives import serialization
import pycryptodome.cipher as cipher
import pycryptodome.public_key.rsa as rsa

class ClientHomomorphicEncryption:
    def __init__(self, private_key):
        self.private_key = serialization.load_pem_private_key(private_key, password=None)

    def encrypt(self, data):
        public_key = self.private_key.public_key()
        cipher_suite = cipher.PKCS7 Padding()
        encryptor = public_key.encryptor()
        encrypted_data = encryptor.update(data.encode('utf-8')) + encryptor.finalize()
        return encrypted_data

    def homomorphic_operation(self, encrypted_data, operation):
        result = int.from_bytes(encrypted_data, 'big') + int(operation)
        return self.encrypt(result.to_bytes(4, 'big'))

client_private_key = b'your_client_private_key'
client = ClientHomomorphicEncryption(client_private_key)

data = 'Hello, World!'
encrypted_data = client.encrypt(data.encode('utf-8'))
print(encrypted_data)

operation = 5
result_encrypted_data = client.homomorphic_operation(encrypted_data, operation)
print(result_encrypted_data)

# For demonstration purposes only:
server_public_key = b'your_client_public'
b'your_client<b'
server  = ServerHomomorphicEncryption(server_public_key)
decrypted_data = server.decrypt(result_encrypted_data)
print(decrypted_data.decode('utf-8'))