from cryptography.hazmat.primitives import serialization
import pycryptodome.cipher as cipher
import pycryptodome.public_key.rsa as rsa

class ServerHomomorphicEncryption:
    def __init__(self, public_key):
        self.public_key = serialization.load_pem_public_key(public_key)

    def decrypt(self, encrypted_data):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        return decrypted_data.decode('utf-8')

    def homomorphic_operation(self, encrypted_data, operation):
        result = int.from_bytes(encrypted_data, 'big') + int(operation)
        return self.encrypt(result.to_bytes(4, 'big'))

    def encrypt(self, data):
        cipher_suite = cipher.PKCS7 Padding()
        encryptor = self.public_key.encryptor()
        encrypted_data = encryptor.update(data.encode('utf-8')) + encryptor.finalize()
        return encrypted_data

server_public_key = b'your_server_public_key'
server = ServerHomomorphicEncryption(server_public_key)

data = 'Hello, World!'
encrypted_data = server.encrypt(data.encode('utf-8'))
print(encrypted_data)

operation = 5
result_encrypted_data = server.homomorphic_operation(encrypted_data, operation)
print(result_encrypted_data)

decrypted_data = server.decrypt(result_encrypted_data)
print(decrypted_data.decode('utf-8'))