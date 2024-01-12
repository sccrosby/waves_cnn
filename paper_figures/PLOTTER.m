classdef PLOTTER
    properties(Constant)
        % WW3 style attributes
        ww3_color = [0. 0. 0.]
        ww3_adj_color = [.5 .5 .5]
        
        % Swell style attributes
        ww3_d_line = '--'
        swell_line = '-'
        
        % Linewidth
        lw = 1.0
        
    end
    
   
    methods(Static)
        
        %%%%%%%%%% Inter hour wave height (Hs)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function plot_inter_hr_hs_swell(self, buoy_6, buoy_12, buoy_24, buoy_48, mycolors)
            
            % Plot WW3
            plot([1 24],[buoy_6.hs_sw_rmse_ww3 buoy_6.hs_sw_rmse_ww3], self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)
            plot([1 24],[buoy_6.hs_sw_rmse_ww3_br buoy_6.hs_sw_rmse_ww3_br], self.ww3_d_line,'Color',self.ww3_adj_color,'LineWidth',self.lw)
            
            % Plot Hours
            plot(buoy_6.hs_sw_rmse_pred,  self.swell_line, 'Color', mycolors(1,:),'LineWidth',self.lw)
            plot(buoy_12.hs_sw_rmse_pred, self.swell_line, 'Color', mycolors(2,:),'LineWidth',self.lw)
            plot(buoy_24.hs_sw_rmse_pred, self.swell_line, 'Color', mycolors(3,:),'LineWidth',self.lw)
            plot(buoy_48.hs_sw_rmse_pred, self.swell_line, 'Color', mycolors(4,:),'LineWidth',self.lw)
            
            set(gca,'XTick',0:6:24)
            set(gca,'XTickLabel',[])
            
            xlim([0 25]);
            
            grid on
            box on
        end
        
        function plot_inter_hr_hs_seas(self, buoy_6, buoy_12, buoy_24, buoy_48, mycolors)
            % WW3 lines
            plot([1 24],[buoy_6.hs_ss_rmse_ww3 buoy_6.hs_ss_rmse_ww3], self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)
            plot([1 24],[buoy_6.hs_ss_rmse_ww3_br buoy_6.hs_ss_rmse_ww3_br], self.ww3_d_line,'Color',self.ww3_adj_color,'LineWidth',self.lw)
            
            % seas
            plot(buoy_6.hs_ss_rmse_pred,  self.swell_line, 'Color', mycolors(1,:),'LineWidth',self.lw)
            plot(buoy_12.hs_ss_rmse_pred, self.swell_line, 'Color', mycolors(2,:),'LineWidth',self.lw)
            plot(buoy_24.hs_ss_rmse_pred, self.swell_line, 'Color', mycolors(3,:),'LineWidth',self.lw)
            plot(buoy_48.hs_ss_rmse_pred, self.swell_line, 'Color', mycolors(4,:),'LineWidth',self.lw)
            
            set(gca,'XTick',0:6:24)
            set(gca,'XTickLabel',[])
            
            xlim([0 25]);
            
            grid on
            box on
            
        end
        
        %%%%%%%%%% Inter hour wave direction (theta)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function plot_inter_hr_theta_swell(self, buoy_6, buoy_12, buoy_24, buoy_48, mycolors, num_hours)
            % WW3 lines
            plot(1:num_hours,buoy_6.rmse_ww3_SW*ones(1,num_hours),self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)

            % Swells
            plot(1:num_hours,buoy_6.rmse_pred_SW,self.swell_line,'Color',mycolors(1,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_12.rmse_pred_SW,self.swell_line,'Color',mycolors(2,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_24.rmse_pred_SW,self.swell_line,'Color',mycolors(3,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_48.rmse_pred_SW,self.swell_line,'Color',mycolors(4,:),'LineWidth',self.lw);
            
            set(gca,'XTick',0:6:24)
            set(gca,'XTickLabel',[])
            
            xlim([0 25]);

            grid on
            box on
        end
        
        function plot_inter_hr_theta_seas(self, buoy_6, buoy_12, buoy_24, buoy_48, mycolors, num_hours)
            % WW3 lines
            plot(1:num_hours,buoy_6.rmse_ww3_SS*ones(1,num_hours),self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)

            % seas
            plot(1:num_hours,buoy_6.rmse_pred_SS,self.swell_line,'Color',mycolors(1,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_12.rmse_pred_SS,self.swell_line,'Color',mycolors(2,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_24.rmse_pred_SS,self.swell_line,'Color',mycolors(3,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_48.rmse_pred_SS,self.swell_line,'Color',mycolors(4,:),'LineWidth',self.lw);
            
            set(gca,'XTick',0:6:24)
            
            xlim([0 25]);

            grid on
            box on
        end
        
        
        %%%%%%%%%% Inter hour wave period (Tm)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function plot_inter_hr_period_swell(self, buoy_6, buoy_12, buoy_24, buoy_48, mycolors)
              
            % WW3 lines
            plot([1 24],[buoy_6.tm_sw_rmse_ww3 buoy_6.tm_sw_rmse_ww3],self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)
            

            % Swells
            plot(buoy_6.tm_sw_rmse_pred,self.swell_line,'Color',mycolors(1,:),'LineWidth',self.lw)
            plot(buoy_12.tm_sw_rmse_pred,self.swell_line,'Color',mycolors(2,:),'LineWidth',self.lw)
            plot(buoy_24.tm_sw_rmse_pred,self.swell_line,'Color',mycolors(3,:),'LineWidth',self.lw)
            plot(buoy_48.tm_sw_rmse_pred,self.swell_line,'Color',mycolors(4,:),'LineWidth',self.lw)
            
            set(gca,'XTick',0:6:24)
            set(gca,'YTick',0:0.2:0.6);
            
            xlim([0 25])
            
            grid on
            box on
            xlabel('Forecast hour')
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Inter Buoy Plots %%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%% Inter buoy wave height (Hs)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function plot_inter_buoy_hs_swell(self, buoy_1, buoy_2, buoy_3, mycolors)
            
            % Plot WW3
            plot([1 24],[buoy_1.hs_sw_rmse_ww3 buoy_1.hs_sw_rmse_ww3], self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)
            plot([1 24],[buoy_1.hs_sw_rmse_ww3_br buoy_1.hs_sw_rmse_ww3_br], self.ww3_d_line,'Color',self.ww3_adj_color,'LineWidth',self.lw)
            
            % Plot Hours
            plot(buoy_1.hs_sw_rmse_pred,  self.swell_line, 'Color', mycolors(1,:),'LineWidth',self.lw)
            plot(buoy_2.hs_sw_rmse_pred, self.swell_line, 'Color', mycolors(2,:),'LineWidth',self.lw)
            plot(buoy_3.hs_sw_rmse_pred, self.swell_line, 'Color', mycolors(3,:),'LineWidth',self.lw)
            
            set(gca,'XTick',0:6:24)
            set(gca,'XTickLabel',[])
            
            xlim([0 25]);
            
            grid on
            box on
            
        end
        
        function plot_inter_buoy_hs_seas(self, buoy_1, buoy_2, buoy_3, mycolors)
            
            % Plot WW3
            plot([1 24],[buoy_1.hs_ss_rmse_ww3 buoy_1.hs_ss_rmse_ww3], self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)
            plot([1 24],[buoy_1.hs_ss_rmse_ww3_br buoy_1.hs_ss_rmse_ww3_br], self.ww3_d_line,'Color',self.ww3_adj_color,'LineWidth',self.lw)
            
            % Plot Hours
            plot(buoy_1.hs_ss_rmse_pred,  self.swell_line, 'Color', mycolors(1,:),'LineWidth',self.lw)
            plot(buoy_2.hs_ss_rmse_pred, self.swell_line, 'Color', mycolors(2,:),'LineWidth',self.lw)
            plot(buoy_3.hs_ss_rmse_pred, self.swell_line, 'Color', mycolors(3,:),'LineWidth',self.lw)
            
            set(gca,'XTick',0:6:24)
            set(gca,'XTickLabel',[])
            
            xlim([0 25]);
            
            grid on
            box on
            
        end
        
        %%%%%%%%%% Inter buoy wave height (theta)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function plot_inter_buoy_theta_swell(self, buoy_1, buoy_2, buoy_3, mycolors, num_hours)
            
            % WW3 lines
            plot(1:num_hours,buoy_1.rmse_ww3_SW*ones(1,num_hours),self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)

            % Swells
            plot(1:num_hours,buoy_1.rmse_pred_SW,self.swell_line,'Color',mycolors(1,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_2.rmse_pred_SW,self.swell_line,'Color',mycolors(2,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_3.rmse_pred_SW,self.swell_line,'Color',mycolors(3,:),'LineWidth',self.lw);
            
            set(gca,'XTick',0:6:24)
            set(gca,'XTickLabel',[])
            
            xlim([0 25]);
            

            grid on
            box on
        end
        
        function plot_inter_buoy_theta_seas(self, buoy_1, buoy_2, buoy_3, mycolors, num_hours)
            
            % WW3 lines
            plot(1:num_hours,buoy_1.rmse_ww3_SS*ones(1,num_hours),self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)

            % Swells
            plot(1:num_hours,buoy_1.rmse_pred_SS,self.swell_line,'Color',mycolors(1,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_2.rmse_pred_SS,self.swell_line,'Color',mycolors(2,:),'LineWidth',self.lw);
            plot(1:num_hours,buoy_3.rmse_pred_SS,self.swell_line,'Color',mycolors(3,:),'LineWidth',self.lw);
            
            %  set(gca,'XTick',0:6:24)
            
            set(gca,'XTick',0:6:24)
            set(gca,'XTickLabel',[])
            
            xlim([0 25]);



            grid on
            box on
        end
        
        %%%%%%%%%% Inter buoy mean period (tm)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function plot_inter_buoy_period_swell(self, buoy_1, buoy_2, buoy_3, mycolors)
              
            % WW3 lines
            plot([1 24],[buoy_1.tm_sw_rmse_ww3 buoy_1.tm_sw_rmse_ww3],self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)
            

            % Swells
            plot(buoy_1.tm_sw_rmse_pred,self.swell_line,'Color',mycolors(1,:),'LineWidth',self.lw)
            plot(buoy_2.tm_sw_rmse_pred,self.swell_line,'Color',mycolors(2,:),'LineWidth',self.lw)
            plot(buoy_3.tm_sw_rmse_pred,self.swell_line,'Color',mycolors(3,:),'LineWidth',self.lw)
            
            set(gca,'XTick',0:6:24)
            set(gca,'YTick',0:0.2:0.6);
            
            xlim([0 25])
            
            grid on
            box on
            xlabel('Forecast hour')
            
        end
        
        function plot_inter_buoy_period_seas(self, buoy_1, buoy_2, buoy_3, mycolors)
              
            % WW3 lines
            plot([1 24],[buoy_1.tm_ss_rmse_ww3 buoy_1.tm_ss_rmse_ww3],self.swell_line,'Color',self.ww3_color,'LineWidth',self.lw)
            

            % Swells
            plot(buoy_1.tm_ss_rmse_pred,self.swell_line,'Color',mycolors(1,:),'LineWidth',self.lw)
            plot(buoy_2.tm_ss_rmse_pred,self.swell_line,'Color',mycolors(2,:),'LineWidth',self.lw)
            plot(buoy_3.tm_ss_rmse_pred,self.swell_line,'Color',mycolors(3,:),'LineWidth',self.lw)
            
            set(gca,'XTick',0:6:24)
            set(gca,'YTick',0:0.2:0.6);
            
            xlim([0 25])
            
            grid on
            box on
            xlabel('Forecast hour')
            
        end
        
    end
end