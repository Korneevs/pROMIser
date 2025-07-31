import pulp
import itertools
import numpy as np
import pandas as pd

class CampaignOptimizer:
    """
    Оптимизатор рекламного бюджета на год (12 месяцев) с тремя сценариями:
      1) Макс. revenue
      2) Макс. DTB_pred
      3) Сбалансированный (revenue - lambda * disbalance)

    Кампания в каждой категории может идти непрерывно до 2 месяцев подряд.
    Всего запусков за год (новых кампаний) должно быть от 10 до 25.
    Поддерживается «no‐good» срез для перечисления K лучших решений.
    """

    def __init__(self,
                 trp_level=1000,
                 months=12,
                 categories=None,
                 lookup=None):
        """
        trp_levels: список дискретных уровней TRP (500,750,…,6000)
        months: число месяцев в году
        categories: список категорий
        lookup: dict[c][m] = pd.DataFrame по уровням TRP с колонками
                'DTB_pred','SOV','budget','revenue','logcats'
        """
        self.trp_levels = list(range(trp_level, 5501, 250))
        self.months      = months
        self.categories  = categories or []
        self.lookup      = lookup or {}

    def optimize(self,
                 B_tot,
                 b_min=None,
                 min_campaigns=None,
                 max_campaigns=None,
                 b_max=None,
                 scenario=1,
                 every_month=True,
                 rk_sale=True,
                 forbidden_plans=None,
                 fixed_campaigns=None):
        """
        B_tot: общий бюджет
        b_min: {c: мин доля бюджета}
        min_campaigns: {c: мин число запусков}
        b_max: {c: макс доля бюджета}
        scenario: 1=revenue, 2=DTB_pred, 3=balanced
        lambd: штраф в scenario=3
        alpha: веса в смешанном варианте
        forbidden_plans: список списков выбранных (c,m,t) для no-good cuts

        Возвращает:
          result: список кампаний с полями
                  category, start_month, end_month, total_TRP, total_budget,
                  total_revenue, total_DTB, avg_ROMI, SOV_profile
          chosen_x: список триплетов (c,m,t) с x[c][m][t]==1
        """
        model = pulp.LpProblem('CampaignOptimization', pulp.LpMaximize)

        # 1) переменные
        x = pulp.LpVariable.dicts('x',
            (self.categories, range(1, self.months+1), self.trp_levels),
            cat=pulp.LpBinary)
        w = pulp.LpVariable.dicts('w',
            (self.categories, range(1, self.months+1)),
            cat=pulp.LpBinary)
        y = pulp.LpVariable.dicts('y',
            (self.categories, range(1, self.months+1)),
            cat=pulp.LpBinary)
        
        if fixed_campaigns is not None:
            for c in self.categories:
                if c not in fixed_campaigns.index: continue
                for m in range(1, self.months+1):
                    if m not in fixed_campaigns.columns: continue
                    if fixed_campaigns.at[c, m] == 1:
                        model += w[c][m] == 1, f"fixed_{c}_{m}"

        
        if 'Sale' in self.categories and rk_sale:
            
            model += w['Sale'][11] == 1, "sale_november_active"
            model += pulp.lpSum(x['Sale'][11][t] for t in self.trp_levels) == 1, "sale_november_exactly_one_trp"

            model += w['Sale'][2] == 1, "sale_february_active"
            model += pulp.lpSum(x['Sale'][2][t] for t in self.trp_levels) == 1, "sale_february_exactly_one_trp"


        for c, m, t in itertools.product(self.categories, range(1, self.months+1), self.trp_levels):
            if self.lookup.get(c, {}).get(m, pd.DataFrame()).at[t, 'SOV'] < 0.13:
                model += x[c][m][t] == 0, f"sov_min_{c}_{m}_{t}"

        # 2) SP-Tires: только месяцы 3,4,10,11
        if 'SP-Tires' in self.categories:
            allowed = {3,4,10,11}
            for m in range(1, self.months+1):
                if m not in allowed:
                    model += w['SP-Tires'][m] == 0
                    
        if 'HL-Furniture' in self.categories:
            allowed = {8,9,10,11,12}
            for m in range(1, self.months+1):
                if m not in allowed:
                    model += w['HL-Furniture'][m] == 0

        # 3) связь x->w
        for c,m in itertools.product(self.categories, range(1,self.months+1)):
            model += pulp.lpSum(x[c][m][t] for t in self.trp_levels) == w[c][m]

        # 4) определение стартов y
        for c in self.categories:
            model += y[c][1] == w[c][1]
            for m in range(2, self.months+1):
                model += y[c][m] <= w[c][m]
                model += y[c][m] <= 1 - w[c][m-1]
                model += y[c][m] >= w[c][m] - w[c][m-1]

        # 5) общее число запусков
        total_campaigns = pulp.lpSum(y[c][m]
                                     for c,m in itertools.product(self.categories, range(1,self.months+1)))
        model += total_campaigns >= 5
        model += total_campaigns <= 19

        # 6) длительность ≤1 месяцев подряд
        for c in self.categories:
            for m in range(1, self.months-1):
                model += w[c][m] + w[c][m+1] <= 1

        # 6.5) мин запуски в категории
        min_campaigns = min_campaigns or {c: 0 for c in self.categories}
        max_campaigns = max_campaigns or {c: self.months for c in self.categories}
        for c in self.categories:
            model += pulp.lpSum(y[c][m] for m in range(1,self.months+1)) \
                    >= min_campaigns.get(c, 0)
            model += pulp.lpSum(y[c][m] for m in range(1,self.months+1)) \
                    <= max_campaigns.get(c, self.months)
            
        M = max(self.trp_levels) * 2  # «большое» число для big-M
        for c in self.categories:
            for m in range(1, self.months+1):
                # TRP в месяце m
                trp_m  = pulp.lpSum(t * x[c][m][t]   for t in self.trp_levels)
                # TRP в месяце m+1 (если он существует)
                trp_m1 = (pulp.lpSum(t * x[c][m+1][t] for t in self.trp_levels)
                          if m < self.months else 0)
                # Если y[c][m]=1 (кампания стартовала здесь), то trp_m+trp_m1 ≤ 5500
                model += trp_m + trp_m1 <= 5500 + M * (1 - y[c][m])

        # 7) no-overlap по logcats (искл. Goods.TiresAndWheels)
        for m in range(1, self.months+1):
            for c1,c2 in itertools.combinations(self.categories,2):
                if m in self.lookup.get(c1,{}) and m in self.lookup.get(c2,{}):
                    cats1 = set(self.lookup[c1][m].iloc[0]['logcats']) - {'Goods.TiresAndWheels'}
                    cats2 = set(self.lookup[c2][m].iloc[0]['logcats']) - {'Goods.TiresAndWheels'}
                    if cats1 & cats2:
                        model += w[c1][m] + w[c2][m] <= 1

        # 8) бюджет
        budget_expr = pulp.lpSum(
            x[c][m][t] * self.lookup[c][m].at[t,'budget']
            for c,m,t in itertools.product(self.categories,
                                           range(1,self.months+1),
                                           self.trp_levels)
        )
        model += budget_expr <= B_tot

        # 8.1) хотя бы одна РК в каждом месяце
        for m in range(1, self.months+1):
            model += pulp.lpSum(w[c][m] for c in self.categories) <= 2, f"max_two_campaigns_month_{m}"
        if every_month == True:
            for m in range(1, self.months+1):
                    model += pulp.lpSum(w[c][m] for c in self.categories) >= 1, f"min_one_campaign_month_{m}"

        # 9) min/max доли по категориям
        b_min = b_min or {c:0.0 for c in self.categories}
        b_max = b_max or {c:B_tot for c in self.categories}
        if sum(b_min.values()) > 0.5 * B_tot:
            raise ValueError("Сумма b_min > 50% бюджета")
        if sum(b_max.values()) < B_tot:
            raise ValueError("Сумма b_max < общий бюджет")
        for c in self.categories:
            catb = pulp.lpSum(
                x[c][m][t] * self.lookup[c][m].at[t,'budget']
                for m,t in itertools.product(range(1,self.months+1), self.trp_levels)
            )
            model += catb >= b_min.get(c,0.0)
            model += catb <= b_max.get(c,B_tot)

        # 10) целевые выражения
        rev_expr = pulp.lpSum(
            x[c][m][t] * self.lookup[c][m].at[t,'revenue']
            for c,m,t in itertools.product(self.categories,
                                           range(1,self.months+1),
                                           self.trp_levels)
        )
        dtb_expr = pulp.lpSum(
            x[c][m][t] * self.lookup[c][m].at[t,'DTB_pred']
            for c,m,t in itertools.product(self.categories,
                                           range(1,self.months+1),
                                           self.trp_levels)
        )
        low_expr = pulp.lpSum(
            x[c][m][t] * self.lookup[c][m].at[t,'low']
            for c,m,t in itertools.product(self.categories,
                                           range(1,self.months+1),
                                           self.trp_levels)
        )

        # 11) постановка цели
        if scenario == 1:
            model += low_expr
        elif scenario == 2:
            model += dtb_expr
        else:
            model += rev_expr

        # 12) no-good cuts
        for idx, plan in enumerate(forbidden_plans or []):
            model += (
                pulp.lpSum(x[c][m][t] for c,m,t in plan)
                <= len(plan) - 1
            ), f"no_good_{idx}"

        # 13) решаем
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # 14) собираем chosen_x
        chosen_x = [
            (c,m,t)
            for c,m,t in itertools.product(self.categories,
                                           range(1,self.months+1),
                                           self.trp_levels)
            if pulp.value(x[c][m][t]) > 0.5
        ]

        # 15) строим привычный result, объединяя до 2 месяцев подряд
        result = []
        for c in self.categories:
            m = 1
            while m <= self.months:
                if any((c,m,t) in chosen_x for t in self.trp_levels):
                    start = m
                    end   = m
                    if end+1 <= self.months and any((c,end+1,t) in chosen_x for t in self.trp_levels):
                        end += 1
                    total_TRP = total_budget = total_revenue = total_dtb = 0
                    sov_profile = []
                    low_CE = []
                    high_CE = []
                    for mm in range(start, end+1):
                        for t in self.trp_levels:
                            if (c,mm,t) in chosen_x:
                                total_TRP     += t
                                total_budget  += self.lookup[c][mm].at[t,'budget']
                                total_revenue += self.lookup[c][mm].at[t,'revenue']
                                total_dtb     += self.lookup[c][mm].at[t,'DTB_pred']
                                sov_profile.append(self.lookup[c][mm].at[t,'SOV'])
                                low_CE.append(self.lookup[c][mm].at[t, 'low'])
                                high_CE.append(self.lookup[c][mm].at[t, 'high']) 
                    avg_romi = (total_revenue/total_budget - 1) if total_budget else 0
                    result.append({
                        'category':     c,
                        'start_month':  start,
                        'end_month':    end,
                        'total_TRP':    total_TRP,
                        'total_budget': total_budget,
                        'total_revenue':total_revenue,
                        'total_DTB':    total_dtb,
                        'avg_ROMI':     avg_romi,
                        'SOV_profile':  np.mean(sov_profile),
                        'low': np.mean(low_CE),
                        'high': np.mean(high_CE)
                    })
                    m = end + 1
                else:
                    m += 1
                

        return result, chosen_x
