from enum import Enum   # Enumモジュール(列挙型を作成するため)
from .state import State

# 描画のタイプを定義する列挙型
class RenderType(Enum):
    LINE = 0   # 線で描画
    DOT = 1    # 点で描画

# ウィンドウのタイプを定義する列挙型
class WindowType(Enum):
    MAIN = 0       # メインウィンドウ
    SEPERATE = 1   # 別ウィンドウ

# 描画オプションを管理するクラス
class RenderOptions:
    def __init__(
            self, 
            name: str,                 # 描画オプションの名前
            color: tuple,              # 描画の色を指定するタプル(R,G,B)
            window_type: WindowType,   # ウィンドウのタイプ(MAIN,SEPARATE)
            render_type: RenderType,   # 描画のタイプ(LINE,DOT)
            min: float,                # 値の最小値
            max: float,                # 値の最大値
            value: float = None,       # 現在の値(デフォルト：None)
        ):
        # コンストラクタで各プロパティを初期化
        self.name = name                # 名前を設定
        self.color = color              # 色を設定
        self.window_type = window_type  # ウィンドウのタイプを設定
        self.render_type = render_type  # 描画タイプを設定
        self.min = min                  # 最小値を設定
        self.max = max                  # 最大値を設定
        self.value = value              # 現在の値を設定

    # 現在のRenderOptionsのコピーを作成するメソッド
    def copy(self):
        return RenderOptions(
            name=self.name,                # 名前をコピー
            color=self.color,              # 色をコピー
            window_type=self.window_type,  # ウィンドウタイプをコピー
            render_type=self.render_type,  # 描画タイプをコピー
            min=self.min,                  # 最小値をコピー
            max=self.max,                  # 最大値をコピー
            value=self.value               # 現在の値をコピー
        )

# カラーテーマを定義するクラス
class ColorTheme:
    black = (0, 0, 0)             # 黒色のRGB値
    white = (255, 255, 255)       # 白色のRGB値
    red = (255, 10, 0)            # 赤色のRGB値
    lightblue = (100, 100, 255)   # ライトブルーのRGB値
    green = (0, 240, 0)           # 緑色のRGB値

    # グラフやインターフェイスの要素に使用する色を指定
    background = black                 # 背景色を黒に設定
    up_candle = green                  # 上昇キャンドルの色を緑に設定
    down_candle = red                  # 降下キャンドルの色を赤に設定
    wick = white                       # ロウソクの芯の色を白に設定
    text = white                       # テキストの色を白に設定
    buy = green                        # 購入を示す色を緑に設定
    sell = red                         # 売却を示す色を赤に設定
    font = 'Noto Sans'                # 使用するフォントを設定
    font_ratio = 0.02                 # フォントサイズの比率を設定（例: 全体のサイズに対する割合
    
# メインウィンドウの設定を管理するクラス
class MainWindow:
    def __init__(
            self, 
            width: int,                    # ウィンドウの幅
            height: int,                   # ウィンドウの高さ
            top_offset: int,               # 上部のオフセット(余白)
            bottom_offset: int,            # 下部のオフセット(余白)
            window_size: int,              # キャンドルの数
            candle_spacing,                # キャンドル間のスペーシング
            font_ratio: float=0.02,        # フォントサイズの比率(デフォルト：0.02)
            spacing_ratio: float=0.02,     # スペーシングの比率(デフォルト：0.02)
            split_offset: int=0            # 分割オフセット(デフォルト：0)
        ):
        # コンストラクタで各プロパティを初期化
        self.width = width                    # ウィンドウの幅を設定
        self.height = height                  # ウィンドウの高さを設定
        self.top_offset = top_offset          # 上部オフセットを設定
        self.bottom_offset = bottom_offset    # 下部オフセットを設定
        self.window_size = window_size        # ウィンドウ内のキャンドル数を設定
        self.candle_spacing = candle_spacing  # キャンドル間のスペースを設定
        self.font_ratio = font_ratio          # フォントサイズの比率を設定
        self.spacing_ratio = spacing_ratio    # スペーシングの比率を設定
        self.split_offset = split_offset      # 分割オフセットを設定

        self.seperate_window_ratio = 0.15     # 別ウィンドウの比率を設定(デフォルト：0.15)

    @property
    # フォントサイズを計算するプロパティ
    def font_size(self):
        # ウィンドウの高さに基づいてフォントサイズを計算
        return int(self.height * self.font_ratio)

    @property
    # キャンドルの幅を計算するプロパティ
    def candle_width(self):
        # 幅をキャンドル数で割り、スペーシングを引く
        return self.width // self.window_size - self.candle_spacing
    
    @property
    # チャートの高さを計算するプロパティ
    def chart_height(self):
        # 高さからオフセットを引いた値を返す
        return self.height - (2 * self.top_offset + self.bottom_offset)
    
    @property
    # スペーシングを計算するプロパティ
    def spacing(self):
        # 高さに基づいてスペーシングを計算
        return int(self.height * self.spacing_ratio)
    
    @property
    # ウィンドウの形状を返すプロパティ
    def screen_shape(self):
        # 幅と高さをタプルで返す
        return (self.width, self.height)
    
    @screen_shape.setter
    # ウィンドウの形状を設定するセッター
    def screen_shape(self, value: tuple):
        # 新しい幅と高さを設定
        self.width, self.height = value

    # 価格をウィンドウの座標にマッピングするメソッド
    def map_price_to_window(self, price: float, max_low: float, max_high: float):
        max_range = max_high - max_low    # 最大値と最小値の範囲を計算
        height = self.chart_height - self.split_offset - self.bottom_offset - self.top_offset * 2  # チャートの高さを計算
        value = int(height - (price - max_low) / max_range * height) + self.top_offset  # 価格を高さにマッピング
        return value   # マッピングされた値を返す
    
    # 値を別ウィンドウの座標にマッピングするメソッド
    def map_to_seperate_window(self, value: float, min: float, max: float):
        # 分割オフセットを計算
        self.split_offset = int(self.height * self.seperate_window_ratio)
        max_range = max - min   # 最大値と最小値の範囲を計算
        new_value = int(self.split_offset - (value - min) / max_range * self.split_offset)  # 値を新しい高さにマッピング
        height = self.chart_height - self.split_offset + new_value  # マッピングされた高さを計算

        return height   # 計算された高さを返す

# Pygameを使用して描画を行うクラス
class PygameRender:
    def __init__(
            self,
            window_size: int=100,        # ウィンドウ内のキャンドル数(デフォルト：100)
            screen_width: int=1440,      # スクリーンの幅(デフォルト：1440pkl)
            screen_height: int=1080,     # スクリーンの高さ(デフォルト：1080pkl)
            top_offset: int=25,          # 上部オフセット(余白)(デフォルト：25pkl)
            bottom_offset: int=25,       # 下部オフセット(余白)(デフォルト：25pkl)
            candle_spacing: int=1,       # キャンドル間のスペーシング(デフォルト：1)
            color_theme = ColorTheme(),  # カラーテーマ(デフォルト：ColorThemeインスタンス)
            frame_rate: int=30,          # フレームレート(デフォルト：30fps)
            render_balance: bool=True,   # バランスを描画するかどうか(デフォルト：True)
        ):
        # Pygameウィンドウの設定を初期化
        self.screen_width = screen_width       # スクリーンの幅を設定
        self.screen_height = screen_height     # スクリーンの高さを設定
        self.top_offset = top_offset           # 上部オフセットを設定
        self.bottom_offset = bottom_offset     # 下部オフセットを設定
        self.candle_spacing = candle_spacing   # キャンドル間のスペーシングを設定
        self.window_size = window_size         # ウィンドウ内のキャンドル数を設定
        self.color_theme = color_theme         # カラーテーマを設定
        self.frame_rate = frame_rate           # フレームレートを設定
        self.render_balance = render_balance   # バランス描画の設定を保存

        # MainWindowクラスのインスタンスを作成
        self.mainWindow = MainWindow(
            width=self.screen_width,
            height=self.screen_height,
            top_offset=self.top_offset,
            bottom_offset=self.bottom_offset,
            window_size=self.window_size,
            candle_spacing=self.candle_spacing,
            font_ratio=self.color_theme.font_ratio  # カラーテーマからフォント比率を所得
        )

        self._states = []  # 状態を保存するリストを初期化

        try:
            import pygame
            self.pygame = pygame  # インポートしたPygameをインスタンス変数に保存
        except ImportError:
            # Pygameがインストールされていない場合、エラーを表示
            raise ImportError('Please install pygame (pip install pygame)')
        
        self.pygame.init()   # Pygameを初期化
        self.pygame.display.init()    # Pygameのディスプレイを初期化
        self.window = self.pygame.display.set_mode(self.mainWindow.screen_shape, self.pygame.RESIZABLE)  # ウィンドウを作成し、リサイズ可能に設定
        self.clock = self.pygame.time.Clock()   # フレームレートを管理するためのクロックを作成

    #　状態をリセットするメソッド
    def reset(self):
        self._states = []   # 内部状態リストを空にする
    
    def _prerender(func):
        """ 入力データの検証とPygameウィンドウの描写を行うデコーダー"""
        # ラッパー関数：元の関数を装飾し、追加の処理を行う
        def wrapper(self, info: dict, rgb_array: bool=False):
            self._states += info.get('states', [])  # infoから'states'を取得し、内部状態リストに追加

            # 内部状態が空またはウィンドウのピクセルアドレスが無効な場合、何もせずに戻る
            if not self._states or not bool(self.window._pixels_address):
                return

            # Pygameイベントを処理
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.pygame.quit()   # 終了イベントが発生した場合、Pygameを終了
                    return

                # ウィンドウがリサイズされた場合、新しいサイズを設定
                if event.type == self.pygame.VIDEORESIZE:
                    self.mainWindow.screen_shape = (event.w, event.h)

                # スペースバーが押されたら一時停止
                if event.type == self.pygame.KEYDOWN:
                    if event.key == self.pygame.K_SPACE:
                        print('Paused')   # 一時停止メッセージを表示
                        while True:
                            event = self.pygame.event.wait()   # イベントが発生するまで待機
                            if event.type == self.pygame.KEYDOWN:
                                if event.key == self.pygame.K_SPACE:  # スペースキーが押されたら
                                    print('Unpaused')   # 再開メッセージを表示
                                    break    # ループを抜ける
                            if event.type == self.pygame.QUIT:
                                self.pygame.quit()   # 終了イベントが発生した場合、Pygameを終了
                                return
                            
                        # ウィンドウサイズを現在の表示サイズに更新
                        self.mainWindow.screen_shape = self.pygame.display.get_surface().get_size()

            # 元の関数を呼び出し、キャンバスを取得
            canvas = func(self, info)
            # キャンバスをウィンドウサイズにスケーリング
            canvas = self.pygame.transform.scale(canvas, self.mainWindow.screen_shape)
            
            # キャンバスの描画をウィンドウにコピー
            self.window.blit(canvas, canvas.get_rect())
            self.pygame.display.update()       # 画面を更新
            self.clock.tick(self.frame_rate)   # フレームレートを維持

            if rgb_array:  # RGB配列を返すオプションがTrueの場合、キャンパスを3次元配列に変換して返す
                return self.pygame.surfarray.array3d(canvas)

        return wrapper   # ラッパー関数を返す
    
    # 与えられた状態に基づいてロウソク足チャート上に指標を描画します
    def render_indicators(self, state: State, canvas: object, candle_offset: int, max_low: float, max_high: float):   # state:現在の状態を示すobject、canvas:Pygameのcanvasobject、candle_offset:ロウソク足のオフセット位置、max_low,max_high:描画する価格の範囲を定義

        # 最後の2点を線で結ぶ処理
        for i, indicator in enumerate(state.indicators):  # 現在の状態から指標を取得
            for name, render_option in indicator["render_options"].items():  # 各指標の描画オブジェクトを取得

                index = self._states.index(state)  # 現在の状態のインデックスを取得
                if not index:  # インデックスが無効な場合、処理を終了
                    return
                last_state = self._states[index - 1]  # 1つ前の状態を取得

                if render_option.render_type == RenderType.LINE:  # 描画タイプがLINEの場合
                    prev_render_option = last_state.indicators[i]["render_options"][name]  # 前の状態の描画オブジェクトを取得

                    # メインウィンドウの場合
                    if render_option.window_type == WindowType.MAIN:
                        # 現在の値と前の値をウィンドウにマッピング
                        cur_value_map = self.mainWindow.map_price_to_window(render_option.value, max_low, max_high)
                        prev_value_map = self.mainWindow.map_price_to_window(prev_render_option.value, max_low, max_high)
                    # 別ウィンドウの場合
                    elif render_option.window_type == WindowType.SEPERATE:
                        # 現在の値と前の値を別ウィンドウにマッピング
                        cur_value_map = self.mainWindow.map_to_seperate_window(render_option.value, render_option.min, render_option.max)
                        prev_value_map = self.mainWindow.map_to_seperate_window(prev_render_option.value, prev_render_option.min, prev_render_option.max)

                    # 現在の値と前の値を線で結ぶ描画
                    self.pygame.draw.line(canvas, render_option.color, 
                                            (candle_offset - self.mainWindow.candle_width / 2, prev_value_map), 
                                            (candle_offset + self.mainWindow.candle_width / 2, cur_value_map))
                    
                elif render_option.render_type == RenderType.DOT:  # 描画タイプがDOTの場合
                    # メインウィンドウの場合
                    if render_option.window_type == WindowType.MAIN:
                        self.pygame.draw.circle(canvas, render_option.color,
                                                (candle_offset, self.mainWindow.map_price_to_window(render_option.value, max_low, max_high)), 2)
                    # 別ウィンドウの場合
                    elif render_option.window == WindowType.SEPERATE:
                        # 実装されていないエラーを発生させる
                        raise NotImplementedError('Seperate window for indicators is not implemented yet')

    # 与えられた状態に基づいてキャンドルを描画
    def render_candle(self, state: State, canvas: object, candle_offset: int, max_low: float, max_high: float, font: object):    # state:現在の状態を示すobject、canvas:Pygameのcanvasobject、candle_offset:ロウソク足のオフセット位置、max_low,max_high:描画する価格の範囲を定義

        assert isinstance(state, State) == True # stateがStateオブジェクトであることを確認

        # キャンドルの座標を計算
        candle_y_open = self.mainWindow.map_price_to_window(state.open, max_low, max_high)   # 開始価格のＹ座標
        candle_y_close = self.mainWindow.map_price_to_window(state.close, max_low, max_high) # 終了価格のＹ座標
        candle_y_high = self.mainWindow.map_price_to_window(state.high, max_low, max_high)   # 高値のＹ座標
        candle_y_low = self.mainWindow.map_price_to_window(state.low, max_low, max_high)     # 安値のＹ座標

        # キャンドルの色を決定
        if state.open < state.close:   # 開始価格が終了価格より低い場合
            # 上昇キャンドル(陽線)
            candle_color = self.color_theme.up_candle   # 色を上昇キャンドル色に設定
            candle_body_y = candle_y_close              # キャンドルのボディのy座標を終了価格に設定
            candle_body_height = candle_y_open - candle_y_close  # ボディの高さを計算
        else:                          # 開始価格が終了価格より高い場合
            # 下降キャンドル(陰線)
            candle_color = self.color_theme.down_candle   # 色を下降キャンドル色に設定
            candle_body_y = candle_y_open                 # キャンドルのボディのy座標を開始価格に設定
            candle_body_height = candle_y_close - candle_y_open   # ボディの高さを計算

        # キャンドルの線（ウィック）を描画
        self.pygame.draw.line(canvas, self.color_theme.wick, 
                              (candle_offset + self.mainWindow.candle_width // 2, candle_y_high), 
                              (candle_offset + self.mainWindow.candle_width // 2, candle_y_low))

        # キャンドルのボディを描画
        self.pygame.draw.rect(canvas, candle_color, (candle_offset, candle_body_y, self.mainWindow.candle_width, candle_body_height))

        # 前の状態と比較して購入又は売却アクションが行われたかを判断し、矢印を描画
        index = self._states.index(state)  # 現在の状態のインデックスを取得

        if index > 0:   # インデックスが0より大きい場合(前の状態が存在する場合)
            last_state = self._states[index - 1]   # 1つ前の状態を取得

            # 所有割合が増加した場合
            if last_state.allocation_percentage < state.allocation_percentage:
                # 購入アクションを示す
                candle_y_low = self.mainWindow.map_price_to_window(last_state.low, max_low, max_high)
                # 前の状態の安値をy座標にマッピング
                self.pygame.draw.polygon(canvas, self.color_theme.buy, [  # 購入矢印を描画
                    (candle_offset - self.mainWindow.candle_width / 2, candle_y_low + self.mainWindow.spacing / 2),   # 矢印の底辺の左端
                    (candle_offset - self.mainWindow.candle_width * 0.1, candle_y_low + self.mainWindow.spacing),   # 矢印の上部左
                    (candle_offset - self.mainWindow.candle_width * 0.9, candle_y_low + self.mainWindow.spacing)   # 矢印の上部右
                    ])
                
                # キャンドルの下にアカウントの値をラベルとして追加
                if self.render_balance:   # バランス描画が有効な場合
                    text = str(int(last_state.account_value))  # アカウントの値を整数として文字列に変換
                    buy_label = font.render(text, True, self.color_theme.text)  # テキストを描画
                    label_width, label_height = font.size(text)   # テキストのサイズを取得
                    canvas.blit(buy_label, (candle_offset - (self.mainWindow.candle_width + label_width) / 2, candle_y_low + self.mainWindow.spacing))  # ラベルをキャンバスに描画

            # 所有割合が減少した場合
            elif last_state.allocation_percentage > state.allocation_percentage:
                # 売却アクションを示す
                candle_y_high = self.mainWindow.map_price_to_window(last_state.high, max_low, max_high)   # 前の状態の高値をy座標にマッピング
                self.pygame.draw.polygon(canvas, self.color_theme.sell, [  # 売却矢印を描画
                    (candle_offset - self.mainWindow.candle_width / 2, candle_y_high - self.mainWindow.spacing / 2),    # 矢印の底辺の左端
                    (candle_offset - self.mainWindow.candle_width * 0.1, candle_y_high - self.mainWindow.spacing),     # 矢印の上部左
                    (candle_offset - self.mainWindow.candle_width * 0.9, candle_y_high - self.mainWindow.spacing)      # 矢印の上部右
                    ])

                # キャンドルの上にアカウントの値をラベルとして追加
                if self.render_balance:    # バランス描画が有効な場合
                    text = str(int(last_state.account_value))  # アカウントの値を整数として文字列に変換
                    sell_label = font.render(text, True, self.color_theme.text)  # テキストを描画
                    label_width, label_height = font.size(text)  # テキストサイズを取得
                    canvas.blit(sell_label, (candle_offset - (self.mainWindow.candle_width + label_width) / 2, candle_y_high - self.mainWindow.spacing - label_height))  # ラベルをキャンバスに描画

    @_prerender  # (このメソッドが呼び出される前に、入力データの検証やウィンドウの更新などの処理を行う)
    # 描画を行うメソッド(info:描画に必要あ情報を含む辞書)
    def render(self, info: dict):
        canvas = self.pygame.Surface(self.mainWindow.screen_shape)  # 描画用のキャンパスを作成
        canvas.fill(self.color_theme.background)  # キャンパスを背景色で塗りつぶす
        
        # 表示するウィンドウサイズ内の最大高値と最小低値を計算(描画に必要な Y 座標を決定)
        max_high = max([state.high for state in self._states[-self.window_size:]])  # 最大高値を取得
        max_low = min([state.low for state in self._states[-self.window_size:]])    # 最小低値を取得

        candle_offset = self.candle_spacing  # キャンドルのオフセットを初期化

        # ラベル用のフォントを設定
        font = self.pygame.font.SysFont(self.color_theme.font, self.mainWindow.font_size)

        # 最新のウィンドウサイズに基づいて、各状態に対してキャンドルを描画
        for state in self._states[-self.window_size:]:

            # 指標を描画
            self.render_indicators(state, canvas, candle_offset, max_low, max_high)

            # キャンドルを描画
            self.render_candle(state, canvas, candle_offset, max_low, max_high, font)

            # 次のキャンドルに移動
            candle_offset += self.mainWindow.candle_width + self.candle_spacing  # オフセットを更新

        # チャートに最大値と最小値のOHLC値を描画
        label_width, label_height = font.size(str(max_low))  # 最小低値のラベルサイズを取得
        label_y_low = font.render(str(max_low), True, self.color_theme.text)  # 最小低値を描画
        canvas.blit(label_y_low, (self.candle_spacing + 5, self.mainWindow.height - label_height * 2))  # 位置を指定してキャンバスの描画

        label_width, label_height = font.size(str(max_low))   # 最大高値のラベルサイズを取得
        label_y_high = font.render(str(max_high), True, self.color_theme.text)  # 最大高値を描画
        canvas.blit(label_y_high, (self.candle_spacing + 5, label_height))  # 位置を指定してキャンバスに描画

        return canvas  # 完成したキャンバスを返す