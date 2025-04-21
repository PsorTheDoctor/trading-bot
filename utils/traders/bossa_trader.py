from os import path
from pathlib import Path
from utils.traders.base_trader import BaseTrader

import socket
import threading
import time
import uuid
import configparser

SOH = '\x01'

class FixSocketClient:
    def __init__(self, host, port, sender_comp_id, target_comp_id, username, password):
        self.host = host
        self.port = port
        self.sender = sender_comp_id
        self.target = target_comp_id
        self.username = username
        self.password = password
        self.socket = None
        self.msg_seq_num = 1
        self.recv_thread = None
        self.running = False

    def connect(self):
        self.socket = socket.create_connection((self.host, self.port))
        self.running = True
        # Uruchom wątek do odbioru
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.recv_thread.start()
        # Wyślij logon
        self._send_logon()

    def disconnect(self):
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None

    def _receive_loop(self):
        buffer = b""
        while self.running:
            try:
                data = self.socket.recv(4096)
                if not data:
                    break
                buffer += data
                # FIX uses SOH delimiters; rozdziel komunikaty
                while b"8=FIX" in buffer:
                    # znajdź koniec komunikatu poprzez SUMY kontrolnej
                    end = buffer.find(b"10=")
                    if end == -1 or len(buffer) < end + 7:
                        break
                    msg = buffer[:end+7]
                    buffer = buffer[end+7:]
                    print("[Received FIX]", msg)
            except Exception as e:
                print("[Error in receive]", e)
                break

    def _calc_body_length(self, msg_without_header):
        # długość pól od 35= aż przed 10=
        return len(msg_without_header.encode('ascii'))

    def _calc_checksum(self, full_msg):
        chksum = sum(full_msg.encode('ascii')) % 256
        return f"{chksum:03}"

    def _build_message(self, fields):
        # Header
        header = [f"8=FIX.4.4"]
        # Body fields: include 35=MsgType
        body = []
        body.extend(fields)
        # Construct without body length and checksum
        body_str = SOH.join(body) + SOH
        length = self._calc_body_length(f"35={fields[0].split('=')[1]}{SOH}{body_str}")
        header.append(f"9={length}")
        header.append(f"35={fields[0].split('=')[1]}")
        msg_without_checksum = SOH.join(header) + SOH + body_str
        checksum = self._calc_checksum(msg_without_checksum)
        full_msg = msg_without_checksum + f"10={checksum}" + SOH
        return full_msg

    def _send_raw(self, msg_text):
        print("[Sending FIX]", msg_text)
        self.socket.sendall(msg_text.encode('ascii'))
        self.msg_seq_num += 1

    def _send_logon(self):
        now = time.strftime("%Y%m%d-%H:%M:%S")
        fields = [
            f"A={self.username}",
            f"554={self.password}",
            f"34={self.msg_seq_num}",
            f"49={self.sender}",
            f"56={self.target}",
            f"52={now}",
            # Domyślnie HeartBtInt=30
            "108=30"
        ]
        msg = self._build_message([f"35=A"] + fields)
        self._send_raw(msg)

    def send_order(self, symbol, side, quantity, price):
        # side: 1=Buy, 2=Sell
        now = time.strftime("%Y%m%d-%H:%M:%S")
        cl_ord_id = uuid.uuid4().hex[:8]
        fields = [
            f"11={cl_ord_id}",
            "21=1",                   # HandlInst: Automated
            f"55={symbol}",
            f"54={side}",
            f"60={now}",
            "40=2",                   # OrdType: Limit
            f"44={price}",
            f"38={quantity}",
            f"59=0"                   # TimeInForce: Day
        ]
        msg = self._build_message(["35=D"] + fields + [
            f"34={self.msg_seq_num}",
            f"49={self.sender}",
            f"56={self.target}",
            f"52={now}"
        ])
        self._send_raw(msg)
        return cl_ord_id

    def cancel_order(self, orig_cl_ord_id, symbol, side, quantity):
        now = time.strftime("%Y%m%d-%H:%M:%S")
        cl_ord_id = uuid.uuid4().hex[:8]
        fields = [
            f"41={orig_cl_ord_id}",
            f"11={cl_ord_id}",
            f"55={symbol}",
            f"54={side}",
            f"60={now}",
            f"38={quantity}"
        ]
        msg = self._build_message(["35=F"] + fields + [
            f"34={self.msg_seq_num}",
            f"49={self.sender}",
            f"56={self.target}",
            f"52={now}"
        ])
        self._send_raw(msg)
        return cl_ord_id

class BossaTrader(BaseTrader):
    def __init__(self):
        super().__init__()
        self.socket = None
        self.msg_seq_num = 1
        self.recv_thread = None
        self.running = False
    
    def _load_config(self):
        config_path = path.join(Path(__file__ ).parent.parent.parent, 'bossa.cfg')
        config = configparser.ConfigParser()
        config.read(config_path)
        
        params = config['DEFAULT']
        return params['Host'], int(params['Port']), params['SenderCompID'], params['TargetCompID'], params['Username'], params['Password']
    
    def connect_with_trader(self):
        host, port, sender, target, user, pwd = self._load_config()

        if mt5.initialize(path=terminal_path, login=int(key[0]), password=key[1], server=key[2]):
            print('Connected')
    
    def market_order(self, symbol, vol, buy_sell):
        if buy_sell.lower()[0] == 'b':
            direction = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:
            direction = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': vol,
            'price': price,
            'type': direction,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_RETURN
        }
        status = mt5.order_send(request)
        return status


    def limit_order(self, symbol, vol, buy_sell, pips_away):
        pip_unit = 10 * mt5.symbol_info(symbol).point

        if buy_sell.lower()[0] == 'b':
            direction = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask + pips_away * pip_unit
        else:
            direction = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid - pips_away * pip_unit

        request = {
            'action': mt5.TRADE_ACTION_PENDING,
            'symbol': symbol,
            'volume': vol,
            'price': price,
            'type': direction,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_RETURN
        }
        status = mt5.order_send(request)
        return status
